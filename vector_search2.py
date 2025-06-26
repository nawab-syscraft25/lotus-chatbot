import os
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from product_utils1 import get_product_stock_status
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
PINECONE_HOST = "https://lotus-products-jsy3z1v.svc.aped-4627-b74a.pinecone.io"
PINECONE_INDEX = "lotus-products"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_WORKERS = 20
CACHE_SIZE = 2000
MAX_QUERY_LENGTH = 256
STOCK_CHECK_TIMEOUT = 5
MAX_RESULTS = 4

# Initialize clients and models with lazy loading
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX, host=PINECONE_HOST)
embedding_model = None  # Lazy load
stock_check_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def get_embedding_model():
    """Lazy load the embedding model"""
    global embedding_model
    if embedding_model is None:
        logger.info("Loading SentenceTransformer model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("SentenceTransformer model loaded successfully")
    return embedding_model

# Regex patterns
RAM_PATTERN = re.compile(r"RAM\s*([^)]+)")
COLOR_PATTERN = re.compile(r"(Black|Gold|Silver|Blue)", re.IGNORECASE)
PRICE_RANGE_RE = re.compile(
    r"""
    (?:
        (?P<under>(?:under|below|less\s+than|upto|max|maximum|budget\s+of?))\s*(?P<uval>[0-9k,]+) |
        (?P<above>(?:above|over|more\s+than|greater\s+than|min|minimum))\s*(?P<aval>[0-9k,]+) |
        between\s*(?P<b1>[0-9k,]+)\s*and\s*(?P<b2>[0-9k,]+) |
        (?P<around>(?:around|approximately|about|near|close\s+to))\s*(?P<rval>[0-9k,]+) |
        (?P<exact>₹?\s*(?P<eval>[0-9k,]+)\s*(?:budget|price|cost|rs?\.?))
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

@lru_cache(maxsize=CACHE_SIZE)
def get_cached_embedding(query: str) -> List[float]:
    """Get cached embedding for query"""
    model = get_embedding_model()
    return model.encode([query])[0].tolist()

def _k_to_int(text: str) -> int:
    """Convert text with 'k' suffix to integer"""
    text = text.replace(",", "").lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1_000)
    return int(text)

def extract_price_filter(query: str) -> Optional[dict]:
    """Extract price filter from query with improved pattern matching"""
    query = query.replace("₹", "")
    match = PRICE_RANGE_RE.search(query)
    if not match:
        return None

    try:
        if match.group("under"):
            return {"$lte": _k_to_int(match.group("uval"))}
        if match.group("above"):
            return {"$gte": _k_to_int(match.group("aval"))}
        if match.group("b1") and match.group("b2"):
            return {"$gte": _k_to_int(match.group("b1")), "$lte": _k_to_int(match.group("b2"))}
        if match.group("around"):
            # For "around X", create a range of ±20%
            val = _k_to_int(match.group("rval"))
            margin = int(val * 0.2)
            return {"$gte": val - margin, "$lte": val + margin}
        if match.group("exact"):
            # For exact budget, create a range of ±10%
            val = _k_to_int(match.group("eval"))
            margin = int(val * 0.1)
            return {"$gte": val - margin, "$lte": val + margin}
    except Exception as e:
        logger.error(f"Error extracting price filter: {e}")
        return None

    return None

def parse_price(price_str: str) -> Optional[float]:
    """Parse price string to float"""
    try:
        # Remove all non-numeric characters except decimal point
        clean = re.sub(r"[^\d.]", "", str(price_str))
        if not clean:
            return None
        return float(clean)
    except Exception:
        return None

async def check_stock_status_async(product_link: str) -> tuple:
    """Check stock status asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                stock_check_pool,
                get_product_stock_status,
                product_link
            ),
            timeout=STOCK_CHECK_TIMEOUT
        )
        
        # Ensure we return a proper tuple
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            return tuple(result)
        else:
            logger.warning(f"Unexpected result format from get_product_stock_status: {result}")
            return False, None, "Invalid result format"
            
    except asyncio.TimeoutError:
        logger.warning(f"Timeout checking stock for: {product_link}")
        return False, None, "Timeout checking stock status"
    except Exception as e:
        logger.error(f"Error checking stock for {product_link}: {e}")
        return False, None, str(e)

def extract_features(text: str) -> List[str]:
    """Extract features from product text"""
    features = []
    if not text:
        return features

    if ram_match := RAM_PATTERN.search(text):
        features.append(f"RAM {ram_match.group(1).strip()}")
    if color_match := COLOR_PATTERN.search(text):
        features.append(color_match.group(1))
    if "5G" in text.upper():
        features.append("5G Connectivity")

    return features

async def process_product_match(match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single product match"""
    try:
        metadata = match.get("metadata", {})
        if not metadata:
            return None

        product_link = metadata.get("product_link", "")
        if not product_link:
            return {
                "name": metadata.get("product_name", ""),
                "price": metadata.get("product_mrp", ""),
                "brand": metadata.get("mpn", ""),
                "link": "",
                "features": extract_features(metadata.get("text", "")),
                "score": match.get("score", 0),
                "in_stock": False,
                "stock_status": "❌ Link Not Available",
                "stock_message": "Product link not available",
                "first_image": "Image not available"
            }

        # Check stock status with proper error handling
        try:
            stock_result = await check_stock_status_async(product_link)
            if isinstance(stock_result, tuple) and len(stock_result) >= 2:
                is_in_stock, first_image = stock_result[0], stock_result[1]
            else:
                # Fallback if the result is not as expected
                logger.warning(f"Unexpected stock result format: {stock_result}")
                is_in_stock, first_image = False, None
        except Exception as e:
            logger.error(f"Error checking stock for {product_link}: {e}")
            is_in_stock, first_image = False, None

        return {
            "name": metadata.get("product_name", ""),
            "price": metadata.get("product_mrp", ""),
            "brand": metadata.get("mpn", ""),
            "link": product_link,
            "features": extract_features(metadata.get("text", "")),
            "score": match.get("score", 0),
            "in_stock": is_in_stock,
            "stock_status": "✅ In Stock" if is_in_stock else "❌ Out of Stock",
            "stock_message": "" if is_in_stock else "Product not available online!\nVisit your nearest store to check for best offline deals.",
            "first_image": first_image or "Image not available"
        }
    except Exception as e:
        logger.error(f"Error processing product match: {e}")
        logger.error(f"Match data: {match}")
        return None

async def search_vector_db_async(query: str, top_k: int = 8) -> Dict[str, Any]:
    """Search vector database asynchronously"""
    try:
        # Get embedding vector
        vec = get_cached_embedding(query[:MAX_QUERY_LENGTH])
        
        # Extract price filter
        price_filter = extract_price_filter(query)
        
        # Query Pinecone without price filtering first
        # We'll apply price filtering in application code
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: index.query(
                vector=vec,
                top_k=top_k * 2,  # Get more results to filter from
                include_metadata=True
            ),
        )
        
        # Process all matches
        tasks = [process_product_match(match) for match in response.get("matches", [])]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing match: {result}")
                continue
            if result is not None:
                valid_results.append(result)

        # Apply price filtering in application code
        if price_filter:
            logger.info(f"Applying price filter: {price_filter}")
            filtered_results = []
            for result in valid_results:
                price = parse_price(result.get('price', ''))
                if price is not None:
                    # Check if price is within range
                    in_range = True
                    if "$lte" in price_filter and price > price_filter["$lte"]:
                        in_range = False
                    if "$gte" in price_filter and price < price_filter["$gte"]:
                        in_range = False
                    
                    if in_range:
                        filtered_results.append(result)
                else:
                    # If we can't parse the price, include it anyway
                    filtered_results.append(result)
            
            valid_results = filtered_results
            logger.info(f"After price filtering: {len(valid_results)} results")

        # Sort results: in-stock first, then by relevance score
        sorted_results = sorted(
            valid_results,
            key=lambda x: (not x["in_stock"], -x["score"]),
            reverse=False
        )

        return {
            "type": "general_search",
            "results": sorted_results[:MAX_RESULTS]
        }
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return {"type": "general_search", "results": []}

def search_vector_db(query: str, top_k: int = 8) -> Dict[str, Any]:
    """Synchronous wrapper for vector search"""
    try:
        # Use a new event loop for this thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_async_search, query, top_k)
            return future.result()
    except Exception as e:
        logger.error(f"Error in search_vector_db: {e}")
        return {"type": "general_search", "results": []}

def _run_async_search(query: str, top_k: int) -> Dict[str, Any]:
    """Helper function to run async search in a new event loop"""
    try:
        return asyncio.run(search_vector_db_async(query, top_k))
    except Exception as e:
        logger.error(f"Error in _run_async_search: {e}")
        return {"type": "general_search", "results": []}

# Pre-load the model on startup to avoid delays during first query
def preload_model():
    """Pre-load the embedding model"""
    logger.info("Pre-loading SentenceTransformer model...")
    get_embedding_model()
    logger.info("Model pre-loading completed")

if __name__ == "__main__":
    # Pre-load model when running directly
    preload_model()
    
    async def test_search():
        test_queries = [
            ("tv under 10000", "Should return TVs priced ≤ ₹10,000"),
            ("smartphone below ₹15000", "Should return smartphones ≤ ₹15,000"),
            ("laptop between 30000 and 50000", "Should return laptops ₹30k-50k"),
            ("mobile above 20000", "Should return mobiles ≥ ₹20,000"),
            ("headphones under 5000", "Should return headphones ≤ ₹5,000"),
            ("camera between 20000 and 40000", "Should return cameras ₹20k-40k")
        ]

        for query, description in test_queries:
            print(f"\n[Testing: '{query}']")
            print(f"Description: {description}")
            
            # Extract expected price range
            price_filter = extract_price_filter(query)
            if price_filter:
                if "$lte" in price_filter:
                    print(f"Expected max price: ₹{price_filter['$lte']}")
                if "$gte" in price_filter:
                    print(f"Expected min price: ₹{price_filter['$gte']}")
                if "$lte" in price_filter and "$gte" in price_filter:
                    print(f"Expected price range: ₹{price_filter['$gte']} - ₹{price_filter['$lte']}")
            
            start_time = time.time()
            results = await search_vector_db_async(query)
            duration = time.time() - start_time
            print(f"Query processed in {duration:.2f} seconds")
            
            if not results.get("results"):
                print("❌ No results found")
                continue
                
            all_in_range = True
            for i, result in enumerate(results.get("results", [])):
                price = parse_price(result['price'])
                if price is None:
                    print(f"⚠️ Could not parse price: {result['price']}")
                    continue
                
                print(f"\n{i+1}. {result['name']}")
                print(f"   Price: ₹{result['price']} (parsed: {price})")
                print(f"   Stock: {result['stock_status']}")
                print(f"   Features: {', '.join(result['features'])}")
                print(f"   Score: {result['score']:.4f}")
                
                # Verify price is within expected range
                if price_filter:
                    if "$lte" in price_filter and price > price_filter["$lte"]:
                        print(f"❌ Price exceeds max (₹{price} > ₹{price_filter['$lte']})")
                        all_in_range = False
                    if "$gte" in price_filter and price < price_filter["$gte"]:
                        print(f"❌ Price below min (₹{price} < ₹{price_filter['$gte']})")
                        all_in_range = False
            
            if price_filter and all_in_range and results.get("results"):
                print("\n✅ All results are within the expected price range")
            elif price_filter and not all_in_range:
                print("\n❌ Some results are outside the expected price range")
            
            print("-" * 80)

    asyncio.run(test_search())