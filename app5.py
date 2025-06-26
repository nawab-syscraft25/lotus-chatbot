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
import re
from concurrent.futures import ThreadPoolExecutor
import joblib
from sentence_transformers import SentenceTransformer
import httpx  
import logging
import json


from intent_classifier import classify_query, is_product_query
from vector_search2 import search_vector_db, parse_price
from utils.Decide_action_with_llm import decide_action_with_llm,generate_conversation_response
from utils.Search_lotus_products import search_lotus_products, extract_json_from_string
from utils.generate_llm import generate_with_llm, run_in_threadpool
from utils.Generate_fallback import generate_fallback_response
from utils.Product_prompt import build_product_prompt, refine_query_with_llm
from utils.helper import extract_price_filter

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Lotus Shopping Assistant")

is_shutting_down = False

# === CORS Config ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-API-Key"],
    max_age=3600,
)

# === Static Files and Templates ===
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

API_KEY = "nawabkhan"
BUFFER_MEMORY_SIZE = 4
LLM_TIMEOUT = 8
MAX_WORKERS = 8
PRODUCT_PROCESS_LIMIT = 3  
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# === Memory Management ===
class ChatMemory:
    def __init__(self):
        self.memory: Dict[str, List[Dict]] = {}
        self.lock = asyncio.Lock()

    async def get_memory(self, session_id: str) -> List[Dict]:
        async with self.lock:
            return self.memory.get(session_id, [])

    async def add_to_memory(self, session_id: str, message: Dict):
        async with self.lock:
            if session_id not in self.memory:
                self.memory[session_id] = []
            # Maintain fixed memory size
            if len(self.memory[session_id]) >= BUFFER_MEMORY_SIZE:
                self.memory[session_id].pop(0)
            self.memory[session_id].append(message)

chat_memory = ChatMemory()

# === Models ===
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

class SearchRequest(BaseModel):
    search_text: str
    alias: Optional[str] = None

# === Auth Middleware ===

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


OPENAI_MODEL = "gpt-4.1"
# === Startup / Shutdown ===
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Lotus Shopping Assistant with OpenAI...")
    import httpx  
    openai = OpenAI(api_key=OPENAI_API_KEY)
    
    # Test OpenAI API key
    try:
        test_response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logger.info("✅ OpenAI API key is working")
    except Exception as e:
        logger.error(f"❌ OpenAI API key test failed: {str(e)}")
        logger.error("Please check your OPENAI_API_KEY in the .env file")
        raise ValueError(f"OpenAI API key validation failed: {str(e)}")
    
    # Preload the embedding model for faster response times
    try:
        from vector_search2 import preload_model
        await asyncio.get_event_loop().run_in_executor(executor, preload_model)
        logger.info("Embedding model preloaded successfully")
    except Exception as e:
        logger.warning(f"Could not preload embedding model: {e}")
        logger.info("Model will be loaded on first use")

@app.on_event("shutdown")
async def shutdown_event():
    global is_shutting_down
    is_shutting_down = True
    logger.info("Shutting down Lotus Shopping Assistant...")
    executor.shutdown(wait=False)


@lru_cache(maxsize=1000)
def cached_vector_search(query: str) -> Dict:
    """Cached vector search - price filtering is now handled internally"""
    return search_vector_db(query[:200])



async def generate_product_response(products: List[Dict], query: str, history: str) -> str:
    """Generate LLM response for products"""
    if not products:
        return "I couldn't find matching products. Would you like to try different keywords?"
    
    prompt = build_product_prompt(products, query, history)
    response = await generate_with_llm(prompt) or "I'm having trouble generating a response."
    print(f"🔍 LLM Response: {response[:500]}...") 
    return response


async def process_product_query(query: str, intent: str, history: str, session_id: str) -> Dict:
    """Handles all product-related queries for Lotus Electronics"""
    try:
        if intent == "specific_product_info":
            products = await search_lotus_products(query, limit=3)
            # print(products)
            in_stock = [p for p in products if p.get("in_stock")]

            if not in_stock:
                return {
                    "status": "success",
                    "data": {
                        "answer": "This product is currently out of stock. Would you like similar alternatives?",
                        "products": [],
                        "intent": intent
                    }
                }

            # Generate structured JSON response via LLM
            response_text = await generate_product_response(in_stock, query, history)
            parsed_response = extract_json_from_string(response_text)

            if not parsed_response:
                logger.warning("LLM response parsing failed, using fallback response")
                return generate_fallback_response(in_stock, query)

            return {
                "status": "success",
                "data": {
                    "answer": parsed_response.get("data", {}).get("answer", ""),
                    "products": parsed_response.get("data", {}).get("products", []),
                    "intent": intent,
                    "end": parsed_response.get("data", {}).get("end", "")
                }
            }

        else:
            # General product queries
            refined_query = await refine_query_with_llm(query, history)
            print(f"Original query: '{query}'")
            print(f"Refined query: '{refined_query}'")
            
            # Price filtering is now handled automatically within the vector search
            results = await run_in_threadpool(cached_vector_search, refined_query)

            docs = results.get("results", [])
            # print(docs)

            # If no results from vector search, try API search as fallback
            if not docs:
                logger.info("No results from vector search, trying API search...")
                # Use original query for API search, not refined query
                api_results = await search_lotus_products(query, limit=5)  # Get more results to filter from
                print(f"API search results: {api_results}")
                
                # If no results with original query, try refined query as fallback
                if not api_results:
                    logger.info("No results with original query, trying refined query...")
                    api_results = await search_lotus_products(refined_query, limit=5)
                    print(f"API search results with refined query: {api_results}")
                
                # Apply price filtering to API results
                price_filter = extract_price_filter(query)
                if price_filter and api_results:
                    logger.info(f"Applying price filter to API results: {price_filter}")
                    filtered_api_results = []
                    for result in api_results:
                        price = parse_price(result.get('price', ''))
                        if price is not None:
                            # Check if price is within range
                            in_range = True
                            if "$lte" in price_filter and price > price_filter["$lte"]:
                                in_range = False
                            if "$gte" in price_filter and price < price_filter["$gte"]:
                                in_range = False
                            
                            if in_range:
                                filtered_api_results.append(result)
                        else:
                            # If we can't parse the price, include it anyway
                            filtered_api_results.append(result)
                    
                    docs = filtered_api_results
                    logger.info(f"After price filtering API results: {len(docs)} results")
                else:
                    docs = api_results

            if not docs:
                return {
                    "status": "success",
                    "data": {
                        "answer": "No matching products found. Try different keywords?",
                        "products": [],
                        "intent": intent
                    }
                }

            response_text = await generate_product_response(docs, query, history)
            parsed_response = extract_json_from_string(response_text)

            if not parsed_response:
                logger.warning("LLM response parsing failed, using fallback response")
                result = generate_fallback_response(docs, query)
            else:
                result = {
                    "status": "success",
                    "data": {
                        "answer": parsed_response.get("data", {}).get("answer", ""),
                        "products": parsed_response.get("data", {}).get("products", []),
                        "intent": intent,
                        "end": parsed_response.get("data", {}).get("end", "")
                    }
                }

        # Add to memory
        await chat_memory.add_to_memory(session_id, {
            "user": query,
            "bot": result.get("data", {}).get("answer", "I'm sorry, I couldn't process that request.")
        })

        return result

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        return {
            "status": "error",
            "message": "Processing failed"
        }

# === Endpoints ===
@app.post("/ask")
async def ask(question: Question, api_key: str = Depends(verify_api_key)):
    if is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Service unavailable"}
        )
        
    try:
        session_id = question.session_id or str(uuid.uuid4())
        history = await chat_memory.get_memory(session_id)
        history_str = "\n".join(f"User: {msg['user']}\nBot: {msg['bot']}" for msg in history)
        
        # Let LLM decide the action
        decision = await decide_action_with_llm(question.question, history_str)
        print(f"🤖 LLM Decision: {decision['action']} - {decision['reasoning']}")
        
        if decision["action"] == "conversation":
            # Continue the conversation
            conversation_response = await generate_conversation_response(question.question, history_str)
            result = {
                "status": "success",
                "data": {
                    "answer": conversation_response,
                    "products": [],
                    "intent": "conversation",
                    "end": ""
                }
            }
        else:
            # Show products - proceed with product search
            # Classify intent
            intent, _ = await run_in_threadpool(classify_query, question.question)
            
            # Process based on intent
            if is_product_query(intent):
                result = await process_product_query(question.question, intent, history_str, session_id)
            else:
                refined_query = await refine_query_with_llm(question.question, history_str)
                print(f"Original query: '{question.question}'")
                print(f"Refined query: '{refined_query}'")
                
                # Price filtering is now handled automatically within the vector search
                results = await run_in_threadpool(cached_vector_search, refined_query)

                docs = results.get("results", [])
                # print(docs)

                # If no results from vector search, try API search as fallback
                if not docs:
                    logger.info("No results from vector search, trying API search...")
                    # Use original query for API search, not refined query
                    api_results = await search_lotus_products(question.question, limit=5)  # Get more results to filter from
                    print(f"API search results: {api_results}")
                    
                    # If no results with original query, try refined query as fallback
                    if not api_results:
                        logger.info("No results with original query, trying refined query...")
                        api_results = await search_lotus_products(refined_query, limit=5)
                        print(f"API search results with refined query: {api_results}")
                    
                    # Apply price filtering to API results
                    price_filter = extract_price_filter(question.question)
                    if price_filter and api_results:
                        logger.info(f"Applying price filter to API results: {price_filter}")
                        filtered_api_results = []
                        for result in api_results:
                            price = parse_price(result.get('price', ''))
                            if price is not None:
                                # Check if price is within range
                                in_range = True
                                if "$lte" in price_filter and price > price_filter["$lte"]:
                                    in_range = False
                                if "$gte" in price_filter and price < price_filter["$gte"]:
                                    in_range = False
                                
                                if in_range:
                                    filtered_api_results.append(result)
                            else:
                                # If we can't parse the price, include it anyway
                                filtered_api_results.append(result)
                        
                        docs = filtered_api_results
                        logger.info(f"After price filtering API results: {len(docs)} results")
                    else:
                        docs = api_results

                if not docs:
                    result = {
                        "status": "success",
                        "data": {
                            "answer": "I couldn't find specific product information for your query. You can ask me about our product categories, current deals, or general information about Lotus Electronics.",
                            "products": [],
                            "intent": intent
                        }
                    }
                else:
                    response_text = await generate_product_response(docs, question.question, history_str)
                    parsed_response = extract_json_from_string(response_text)

                    if not parsed_response:
                        logger.warning("LLM response parsing failed, using fallback response")
                        result = generate_fallback_response(docs, question.question)
                    else:
                        result = {
                            "status": "success",
                            "data": {
                                "answer": parsed_response.get("data", {}).get("answer", ""),
                                "products": parsed_response.get("data", {}).get("products", []),
                                "intent": intent,
                                "end": parsed_response.get("data", {}).get("end", "")
                            }
                        }

        # Add to memory
        await chat_memory.add_to_memory(session_id, {
            "user": question.question,
            "bot": result.get("data", {}).get("answer", "I'm sorry, I couldn't process that request.")
        })

        return result

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error"}
        )

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 