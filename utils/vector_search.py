import os
import re
import asyncio
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from product_utils1 import get_product_stock_status
import time


# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Environment & Constants ===
load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
MAX_WORKERS = int(os.getenv("VECTOR_MAX_WORKERS", 20))
CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", 2000))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 256))
STOCK_CHECK_TIMEOUT = float(os.getenv("STOCK_CHECK_TIMEOUT", 5))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 4))
PINECONE_HOST = "https://lotus-products-jsy3z1v.svc.aped-4627-b74a.pinecone.io"
PINECONE_INDEX = "lotus-products"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Pinecone Client ===
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX, host=PINECONE_HOST)

# === Embedding Model ===
_embedding_model: Optional[SentenceTransformer] = None

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

# === Embedding Cache ===
@lru_cache(maxsize=CACHE_SIZE)
def get_cached_embedding(query: str) -> List[float]:
    query = query[:MAX_QUERY_LENGTH]
    vec = get_embedding_model().encode([query])[0]
    return vec.tolist()

# === Price Parsing ===
def _k_to_int(txt: str) -> int:
    t = txt.replace(",", "").lower()
    return int(float(t[:-1]) * 1000) if t.endswith("k") else int(t)

PRICE_PATTERN = re.compile(
    r"under\s*(?P<u>[0-9k,]+)|above\s*(?P<a>[0-9k,]+)|between\s*(?P<b1>[0-9k,]+)\s*and\s*(?P<b2>[0-9k,]+)",
    re.IGNORECASE
)

def extract_price_filter(query: str) -> Optional[Dict[str, int]]:
    m = PRICE_PATTERN.search(query)
    if not m:
        return None
    if m.group('u'):
        return {'$lte': _k_to_int(m.group('u'))}
    if m.group('a'):
        return {'$gte': _k_to_int(m.group('a'))}
    if m.group('b1') and m.group('b2'):
        return {'$gte': _k_to_int(m.group('b1')), '$lte': _k_to_int(m.group('b2'))}
    return None

def parse_price(price_str: str) -> Optional[float]:
    num = re.sub(r"[^\d.]", "", price_str)
    return float(num) if num else None

# === Concurrency Controls ===
_stock_semaphore = asyncio.Semaphore(MAX_WORKERS)
stock_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def check_stock(link: str) -> (bool, Optional[str]):
    async with _stock_semaphore:
        try:
            loop = asyncio.get_event_loop()
            in_stock, img = await asyncio.wait_for(
                loop.run_in_executor(stock_executor, get_product_stock_status, link),
                timeout=STOCK_CHECK_TIMEOUT
            )
            return bool(in_stock), img
        except Exception as e:
            logger.warning("Stock check failed for %s: %s", link, e)
            return False, None

# === Feature Extraction ===
RAM_PATTERN = re.compile(r"(\d+GB) RAM", re.IGNORECASE)
COLOR_PATTERN = re.compile(r"(Black|Silver|Blue|Gold)", re.IGNORECASE)

def extract_features(desc: str) -> List[str]:
    feats: List[str] = []
    if ram := RAM_PATTERN.search(desc): feats.append(ram.group(1))
    if col := COLOR_PATTERN.search(desc): feats.append(col.group(1))
    if '5G' in desc.upper(): feats.append('5G')
    return feats

# === Async Vector Search ===
async def search_vector_db_async(query: str, top_k: int = MAX_RESULTS * 2) -> Dict[str, Any]:
    vec = get_cached_embedding(query)
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: index.query(vector=vec, top_k=top_k, include_metadata=True)
    )

    # Process matches concurrently
    tasks = []
    for match in response.get('matches', []):
        meta = match.get('metadata', {})
        link = meta.get('product_link')
        tasks.append(process_match(match.get('score', 0.0), meta, link))

    results = await asyncio.gather(*tasks)

    # Apply price filter
    pf = extract_price_filter(query)
    if pf:
        results = [r for r in results if (p := parse_price(r['price'])) is None or (pf.get('$gte', 0) <= p <= pf.get('$lte', float('inf')))]

    # Sort: in-stock first, then by score descending
    results.sort(key=lambda x: (not x['in_stock'], -x['score']))
    return {'type': 'general_search', 'results': results[:MAX_RESULTS]}

async def process_match(score: float, meta: Dict[str, Any], link: Optional[str]) -> Dict[str, Any]:
    name = meta.get('product_name', '')
    price = meta.get('product_mrp', '')
    desc = meta.get('text', '')
    in_stock, img = await check_stock(link) if link else (False, None)
    return {
        'name': name,
        'price': price,
        'link': link or '',
        'features': extract_features(desc),
        'score': score,
        'in_stock': in_stock,
        'stock_status': '✅ In Stock' if in_stock else '❌ Out of Stock',
        'first_image': img or 'Image not available'
    }

# === Sync Wrapper ===
def search_vector_db(query: str, top_k: int = MAX_RESULTS) -> Dict[str, Any]:
    return asyncio.run(search_vector_db_async(query, top_k))

# === Preload ===
def preload_model():
    get_embedding_model()
