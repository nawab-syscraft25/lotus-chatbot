from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List
import uvicorn
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging

from intent_classifier import classify_query, is_product_query
from utils.vector_search import search_vector_db, parse_price, preload_model
from utils.Decide_action_with_llm import decide_action_with_llm, generate_conversation_response
from utils.Search_lotus_products import search_lotus_products, extract_json_from_string
from utils.generate_llm import generate_with_llm, run_in_threadpool
from utils.Generate_fallback import generate_fallback_response
from utils.Product_prompt import build_product_prompt, refine_query_with_llm
from utils.helper import extract_price_filter

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"
API_KEY = "nawabkhan"
BUFFER_MEMORY_SIZE = 5
MAX_WORKERS = 8
LLM_TIMEOUT = 8  # seconds for LLM calls

# === FastAPI Lifespan and App ===
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def lifespan(app: FastAPI):
    logger.info("Starting Lotus Shopping Assistant...")
    app.state.openai = OpenAI(api_key=OPENAI_API_KEY)

    # Validate API key
    try:
        app.state.openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logger.info("✅ OpenAI API key is valid")
    except Exception as exc:
        logger.error(f"OpenAI validation failed: {exc}")
        raise

    # Preload embedding model
    await asyncio.get_event_loop().run_in_executor(executor, preload_model)
    logger.info("Embedding model preloaded")

    yield

    executor.shutdown(wait=False)
    logger.info("Shutting down Lotus Shopping Assistant")

app = FastAPI(title="Lotus Shopping Assistant", lifespan=lifespan)

# === Middleware & Static Files ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-API-Key"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Chat Memory ===
class ChatMemory:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.store: Dict[str, List[Dict]] = {}
        self.lock = asyncio.Lock()

    async def get(self, session_id: str) -> List[Dict]:
        async with self.lock:
            return self.store.get(session_id, [])

    async def add(self, session_id: str, message: Dict):
        async with self.lock:
            msgs = self.store.setdefault(session_id, [])
            if len(msgs) >= self.buffer_size:
                msgs.pop(0)
            msgs.append(message)

chat_memory = ChatMemory(BUFFER_MEMORY_SIZE)

# === Pydantic Models ===
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

# === Auth Dependency ===
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# === Cached Vector Search ===
cached_vector_search = lru_cache(maxsize=1000)(lambda q: search_vector_db(q[:200]))

# === Product Query Handler ===
async def handle_product_query(query: str, intent: str, history: str, session_id: str) -> Dict:
    try:
        # Fetch products
        products = await (search_lotus_products(query, limit=3) if intent == "specific_product_info" else search_lotus_products(query, limit=5))
        # Filter stock if specific
        if intent == "specific_product_info":
            products = [p for p in products if p.get("in_stock")]
            if not products:
                return {"status": "success", "data": {"answer": "This product is currently out of stock. Would you like similar alternatives?", "products": [], "intent": intent}}

        # If general and empty, try vector search
        if intent != "specific_product_info":
            # Run vector search with timeout after refining query
            refined_query = await asyncio.wait_for(refine_query_with_llm(query, history), timeout=LLM_TIMEOUT)
            try:
                results = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(executor, cached_vector_search, refined_query), timeout=3)
            except asyncio.TimeoutError:
                logger.warning("Vector search timed out, falling back to API search")
                results = {"results": []}
            products = results.get("results", []) or products

        # Apply price filter
        pf = extract_price_filter(query)
        if pf:
            products = [p for p in products if (pr := parse_price(p.get("price", ""))) is None or (pf.get("$gte", 0) <= pr <= pf.get("$lte", float('inf')))]

        if not products:
            return {"status": "success", "data": {"answer": "No matching products found. Try different keywords?", "products": [], "intent": intent}}

        # Build prompt & get LLM response
        prompt = build_product_prompt(products, query, history)
        llm_resp = await asyncio.wait_for(generate_with_llm(prompt), timeout=LLM_TIMEOUT)
        parsed = extract_json_from_string(llm_resp)
        if not parsed:
            return generate_fallback_response(products, query)

        data = parsed.get("data", {})
        # Ensure image key for each product, falling back when missing
        enriched = []
        for p in data.get("products", []):
            orig_image = next((orig.get("image") for orig in products if orig.get("name") == p.get("name")), "")
            img = p.get("image", "")
            if not img or img.lower().startswith("image not available"):
                img = orig_image
            enriched.append({**p, "image": img})
        data["products"] = enriched

        await chat_memory.add(session_id, {"user": query, "bot": data.get("answer", "")})
        return {"status": "success", "data": data}

    except Exception as e:
        logger.error(f"Error in handle_product_query: {e}")
        return {"status": "error", "message": "Processing failed"}

# === /ask Endpoint ===
@app.post("/ask")
async def ask(question: Question, api_key: str = Depends(verify_api_key), request: Request = None):
    session_id = question.session_id or str(uuid.uuid4())
    history = await chat_memory.get(session_id)
    history_str = "\n".join(f"User: {m['user']}\nBot: {m['bot']}" for m in history)

    # Decide action
    decision = await asyncio.wait_for(decide_action_with_llm(question.question, history_str), timeout=LLM_TIMEOUT)
    if decision.get("action") == "conversation":
        answer = await asyncio.wait_for(generate_conversation_response(question.question, history_str), timeout=LLM_TIMEOUT)
        products = []
        intent = "conversation"
    else:
        intent, _ = await asyncio.wait_for(run_in_threadpool(classify_query, question.question), timeout=LLM_TIMEOUT)
        if is_product_query(intent):
            return await handle_product_query(question.question, intent, history_str, session_id)
        # Non-product fallback
        answer = "I couldn't find specific product information."
        products = []

    await chat_memory.add(session_id, {"user": question.question, "bot": answer})
    return {"status": "success", "data": {"answer": answer, "products": products, "intent": intent, "end": ""}}

# === Root Endpoint ===
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === Main ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
