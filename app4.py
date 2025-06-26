from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import asyncio
import signal
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List, Tuple, Any
import uvicorn
from functools import lru_cache
import re
from concurrent.futures import ThreadPoolExecutor
import joblib
from sentence_transformers import SentenceTransformer
import httpx  # Replaced requests with async httpx
import logging


from utils.product_image_extractor import ProductImageExtractor
from intent_classifier import classify_query, is_product_query
from vector_search2 import search_vector_db, parse_price

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
BUFFER_MEMORY_SIZE = 5
LLM_TIMEOUT = 8
MAX_WORKERS = 8
PRODUCT_PROCESS_LIMIT = 3  # Max products to process concurrently

# Initialize thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# === Initialize OpenAI ===
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client with explicit httpx configuration to avoid version conflicts
try:
    openai = OpenAI(api_key=OPENAI_API_KEY)
except TypeError as e:
    if "proxies" in str(e):
        logger.warning("httpx version conflict detected, trying alternative initialization...")
        # Try without explicit httpx configuration
        import httpx
        openai = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=httpx.Client(timeout=30.0)
        )
    else:
        raise e

# OPENAI_MODEL = "gpt-3.5-turbo"  # You can change this to gpt-4 if you have access
OPENAI_MODEL = "gpt-4.1"  # You can change this to gpt-4 if you have access

# === In-Memory Chat Memory ===
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

# === Startup / Shutdown ===
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Lotus Shopping Assistant with OpenAI...")
    
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

# === Optimized Helper Functions ===
def normalize_currency(query: str) -> str:
    """Improved currency normalization with better pattern matching"""
    patterns = [
        (r'\$(\d+(?:\.\d+)?)', r'₹\1'),  # $100 -> ₹100
        (r'(?i)inr\s*(\d+)', r'₹\1'),
        (r'(?i)rs\.?\s*(\d+)', r'₹\1'),
        (r'(?i)rupees?\s*(\d+)', r'₹\1'),
        (r'(?i)under\s+(\d+)', r'under ₹\1')
    ]
    for pattern, replacement in patterns:
        query = re.sub(pattern, replacement, query)
    return query

@lru_cache(maxsize=1000)
def cached_vector_search(query: str) -> Dict:
    """Cached vector search - price filtering is now handled internally"""
    return search_vector_db(query[:200])

def vector_search_with_filter(query: str, price_filter: dict) -> Dict:
    """Vector search with explicit price filter - now deprecated as filtering is internal"""
    # Price filtering is now handled automatically within the vector search
    # This function is kept for backward compatibility
    return search_vector_db(query[:200])

async def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

# === System Prompt ===
SYSTEM_PROMPT = """## Identity
You are the Customer Support AI Agent for Lotus Electronics. Your role is to interact with customers, address their inquiries, and provide assistance with common support topics.

## Conversation-First Approach
- **Always start with conversation**: Engage with customers naturally before showing products
- **Understand needs first**: Ask questions to understand what they're looking for
- **Only show products when appropriate**: Don't immediately jump to product listings
- **Build rapport**: Be friendly, helpful, and professional

## Scope
- Focus on customer inquiries about orders, billing, account issues, and general support
- Help customers find the right products through conversation
- Do not handle advanced technical support or sensitive financial issues
- Redirect or escalate issues outside your expertise to a human agent

## Responsibility
- Initiate interactions with a friendly greeting
- Guide the conversation based on customer needs
- Ask clarifying questions to understand requirements
- Provide accurate and concise information
- Only show products when the customer specifically asks or when you have enough information
- Escalate to a human agent when customer inquiries exceed your capabilities

## Response Style
- Maintain a friendly, clear, and professional tone
- Keep responses conversational and engaging
- Ask follow-up questions to better understand customer needs
- Use buttons for quick replies and easy navigation whenever possible

## Product Recommendations
- **Don't show products immediately**: First understand what the customer needs
- **Ask questions**: "What type of product are you looking for?" "What's your budget?" "Any specific features you need?"
- **Only show products when**: 
  - Customer specifically asks for products
  - You have enough information to make relevant recommendations
  - Customer mentions specific product categories or requirements

## Ability
- Delegate specialized tasks to AI-Associates or escalate to a human when needed

## Guardrails
- **Privacy**: Respect customer privacy; only request personal data if absolutely necessary
- **Accuracy**: Provide verified and factual responses coming from Knowledge Base or official sources. Avoid speculation

## Instructions
- **Greeting**: Start every conversation with a friendly welcome.  
  _Example_: "Hi, welcome to Lotus Electronics Support! How can I help you today?"

- **Conversation Flow**: 
  1. Greet and ask how you can help
  2. Listen to their needs
  3. Ask clarifying questions if needed
  4. Only show products when appropriate
  5. Offer additional assistance

- **Escalation**: When a customer query becomes too complex or sensitive, notify the customer that you'll escalate the conversation to a human agent.  
  _Example_: "I'm having trouble resolving this. Let me get a human agent to assist you further."

- **Closing**: End interactions by confirming that the customer's issue has been addressed.  
  _Example_: "Is there anything else I can help you with today?"
"""

# === Optimized LLM Functions ===
def _call_openai_api(messages: List[Dict]) -> str:
    """Helper function to call OpenAI API"""
    try:
        print(f"🔍 Calling OpenAI API with model: {OPENAI_MODEL}")
        # print(f"🔍 Messages: {messages}")
        
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        # print(f"🔍 OpenAI API Response: {response}")
        
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content.strip()
            # print(f"🔍 Extracted content: {content[:200]}...")
            return content
        else:
            print(f"❌ Invalid response structure: {response}")
            return ""
            
    except Exception as e:
        print(f"❌ OpenAI API call failed: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        logger.error(f"OpenAI API call failed: {str(e)}")
        return ""

async def generate_with_llm(prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = await asyncio.wait_for(
            run_in_threadpool(_call_openai_api, messages),
            timeout=LLM_TIMEOUT
        )
        return response if response else ""
    except (asyncio.TimeoutError, Exception) as e:
        logger.error(f"OpenAI generation failed: {str(e)}")
        return ""

async def refine_query_with_llm(user_query: str, chat_history: str) -> str:
    """
    Refines the user's product search query to better capture intent and context for vector DB search.
    Uses ₹ for prices and reformulates the query for semantic embedding.
    """
    prompt = f"""
    Goal: Understand the customer's intent and return a concise, info-rich phrase suitable for product search.

    CRITICAL RULES:
    1. ALWAYS preserve the product category/type (TV, smartphone, laptop, AC, etc.)
    2. NEVER remove the product name/category from the query
    3. Use ₹ for prices to maintain consistency
    4. Include key details: product, brand, features, budget, etc.
    5. Extract context from the conversation history if needed
    6. Keep the query focused on what the customer is looking for

    Examples:
    - "Televisions, ₹20000 budget, best deals" → "TV television ₹20000 budget"
    - "Smartphone under 15000" → "smartphone mobile phone ₹15000"
    - "AC around 25000" → "air conditioner AC ₹25000"
    - "Laptop between 30000 and 50000" → "laptop computer ₹30000-50000"
    - "Here are some smartphones under ₹20,000" → "smartphone mobile phone ₹20000"

    Conversation History:
    {chat_history[-800:]}

    Customer Query:
    {user_query}

    Optimized Search Query:"""

    refined = await generate_with_llm(prompt)
    print(f"🔍 Query refinement: '{user_query}' -> '{refined}'")
    return refined if refined else user_query

# === API Configuration ===
LOTUS_API_BASE = "https://portal.lotuselectronics.com/web-api/home"
LOTUS_API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "auth-token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNjg5MzYiLCJpYXQiOjE3NDg5NDc2NDEsImV4cCI6MTc0ODk2NTY0MX0.uZeQseqc6mpm5vkOAmEDgUeWIfOI5i_FnHJRaUBWlMY",
    "content-type": "application/x-www-form-urlencoded",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0"
}

# === Async HTTP Client ===
async_client = httpx.AsyncClient(timeout=10.0)

async def get_product_details(product_id: str) -> Tuple[bool, Dict]:
    """Get product details with async HTTP"""
    try:
        url = f"{LOTUS_API_BASE}/product_detail"
        data = {
            "product_id": product_id,
            "cat_name": f"/product/{product_id}",
            "product_name": f"product-{product_id}"
        }

        response = await async_client.post(url, headers=LOTUS_API_HEADERS, data=data)
        response.raise_for_status()
        result = response.json()
        
        if "data" in result and "product_detail" in result["data"]:
            detail = result["data"]["product_detail"]
            instock = detail.get("instock", "").lower()
            out_of_stock = detail.get("out_of_stock", "0")
            quantity = int(detail.get("product_quantity", "0"))
            
            is_in_stock = (instock == "yes" and out_of_stock == "0" and quantity > 0)
            return is_in_stock, detail
            
        return False, {}
    except Exception as e:
        logger.error(f"Product detail error: {str(e)}")
        return False, {}

def extract_product_category_for_api(query: str) -> str:
    """
    Extract product category from query for API search.
    Returns a clean, simple search term for the Lotus API.
    """
    # Clean the query
    query_lower = query.lower()
    
    # Remove price-related terms and other noise
    price_patterns = [
        r'₹\d+', r'\d+\s*rs?', r'budget', r'price', r'cost', r'under', r'above', 
        r'between', r'around', r'approximately', r'best deals', r'deals', r'offer',
        r'my budget is', r'budget of', r'budget for', r'within budget', r'show me the price of',
        r'what is the price of', r'how much is', r'how much does', r'cost of'
    ]
    
    for pattern in price_patterns:
        query_lower = re.sub(pattern, '', query_lower)
    
    # Check if this looks like a specific product request
    words = query_lower.split()
    
    # If query has 3+ words and contains numbers or specific model indicators, 
    # it's likely a specific product request
    has_numbers = any(re.search(r'\d+', word) for word in words)
    has_model_indicators = any(word in ['pro', 'max', 'ultra', 'plus', 'mini', 'se', 'xl'] for word in words)
    has_brand_indicators = any(word in ['iphone', 'samsung', 'galaxy', 'vivo', 'oppo', 'oneplus', 'xiaomi', 'realme', 'pixel', 'sony', 'lg'] for word in words)
    
    # If it looks like a specific product, return the full product name
    if (len(words) >= 3 and (has_numbers or has_model_indicators)) or has_brand_indicators:
        # Clean up the product name
        product_name = ' '.join([word for word in words if len(word) > 1])
        return product_name.strip()
    
    # Common product categories and their variations
    category_mappings = {
        'tv': ['tv', 'television', 'televisions', 'smart tv', 'led tv', 'oled tv', 'qled tv', '4k tv', 'ultra hd tv'],
        'smartphone': ['smartphone', 'smartphones', 'mobile', 'mobiles', 'phone', 'phones', 'android phone'],
        'laptop': ['laptop', 'laptops', 'notebook', 'notebooks', 'computer', 'pc'],
        'ac': ['ac', 'air conditioner', 'air conditioners', 'split ac', 'window ac', 'cooling'],
        'refrigerator': ['refrigerator', 'refrigerators', 'fridge', 'fridges', 'cooling'],
        'washing machine': ['washing machine', 'washing machines', 'washer', 'laundry'],
        'microwave': ['microwave', 'microwaves', 'oven', 'cooking'],
        'headphones': ['headphones', 'headphone', 'earphones', 'earphone', 'earbuds'],
        'speaker': ['speaker', 'speakers', 'bluetooth speaker', 'sound'],
        'camera': ['camera', 'cameras', 'digital camera', 'photography'],
        'tablet': ['tablet', 'tablets', 'ipad', 'android tablet'],
        'printer': ['printer', 'printers', 'printing'],
        'monitor': ['monitor', 'monitors', 'computer monitor', 'display'],
        'keyboard': ['keyboard', 'keyboards', 'typing'],
        'mouse': ['mouse', 'mice', 'pointing'],
        'router': ['router', 'routers', 'wifi router', 'internet'],
        'power bank': ['power bank', 'power banks', 'powerbank', 'battery'],
        'charger': ['charger', 'chargers', 'mobile charger', 'charging'],
        'cable': ['cable', 'cables', 'usb cable', 'hdmi cable', 'wire'],
        'adapter': ['adapter', 'adapters', 'power adapter', 'connector']
    }
    
    # Find the most relevant category
    for category, keywords in category_mappings.items():
        for keyword in keywords:
            if keyword in query_lower:
                return category
    
    # If no specific category found, try to extract from the beginning of the query
    if words:
        # Check if the first word is a category
        first_word = words[0]
        for category, keywords in category_mappings.items():
            if first_word in keywords:
                return category
        
        # If still no match, return the first meaningful word
        for word in words:
            if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'best', 'good', 'new', 'my', 'is', 'are', 'was', 'were']:
                return word
    
    return query

async def search_lotus_products(query: str, limit: int = 10) -> List[Dict]:
    """Async product search with concurrent detail fetching"""
    try:
        # First, try to search with the specific product name
        specific_query = extract_product_category_for_api(query)
        logger.info(f"API search query: '{query}' -> '{specific_query}'")
        
        # Try the specific product search first
        url = f"{LOTUS_API_BASE}/search_products"
        data = {
            "search_text": specific_query,
            "alias": "",
            "is_brand_search": "0",
            "limit": str(limit),
            "offset": "0",
            "orderby": ""
        }
        
        response = await async_client.post(url, headers=LOTUS_API_HEADERS, data=data)
        response.raise_for_status()
        result = response.json()
        
        products = result.get("data", {}).get("products", [])
        
        # If no results with specific query, try category-based search
        if not products:
            logger.info(f"No results with specific query '{specific_query}', trying category search...")
            
            # Extract category from original query
            category_query = None
            query_lower = query.lower()
            
            # Common product categories
            category_mappings = {
                'smartphone': ['smartphone', 'smartphones', 'mobile', 'mobiles', 'phone', 'phones', 'iphone', 'android'],
                'tv': ['tv', 'television', 'televisions', 'smart tv', 'led tv', 'oled tv', 'qled tv', '4k tv'],
                'laptop': ['laptop', 'laptops', 'notebook', 'notebooks', 'computer', 'pc'],
                'ac': ['ac', 'air', 'conditioner', 'air conditioner', 'air conditioners', 'split ac', 'window ac'],
                'refrigerator': ['refrigerator', 'refrigerators', 'fridge', 'fridges'],
                'washing machine': ['washing machine', 'washing machines', 'washer', 'laundry'],
                'microwave': ['microwave', 'microwaves', 'oven', 'cooking'],
                'headphones': ['headphones', 'headphone', 'earphones', 'earphone', 'earbuds'],
                'speaker': ['speaker', 'speakers', 'bluetooth speaker', 'sound'],
                'camera': ['camera', 'cameras', 'digital camera', 'photography'],
                'tablet': ['tablet', 'tablets', 'ipad', 'android tablet']
            }
            
            for category, keywords in category_mappings.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        category_query = category
                        break
                if category_query:
                    break
            
            if category_query:
                logger.info(f"Trying category search with: '{category_query}'")
                data["search_text"] = category_query
                response = await async_client.post(url, headers=LOTUS_API_HEADERS, data=data)
                response.raise_for_status()
                result = response.json()
                products = result.get("data", {}).get("products", [])
        
        if not products:
            return []
        
        # Process first N products concurrently
        products = products[:PRODUCT_PROCESS_LIMIT]
        tasks = [get_product_details(p["product_id"]) for p in products if "product_id" in p]
        details_results = await asyncio.gather(*tasks)
        
        processed_products = []
        for idx, (is_in_stock, product_detail) in enumerate(details_results):
            if not product_detail:
                continue
                
            p = products[idx]
            
            # Handle features properly
            features = product_detail.get("product_specification", [])
            if isinstance(features, list):
                # Convert feature objects to strings
                feature_strings = []
                for feature in features[:6]:
                    if isinstance(feature, dict):
                        if 'fkey' in feature and 'fvalue' in feature:
                            feature_strings.append(f"{feature['fkey']}: {feature['fvalue']}")
                        elif 'key' in feature and 'value' in feature:
                            feature_strings.append(f"{feature['key']}: {feature['value']}")
                    elif isinstance(feature, str):
                        feature_strings.append(feature)
                features = feature_strings
            
            processed_products.append({
                "name": product_detail.get("product_name", ""),
                "link": f"https://www.lotuselectronics.com/product/{product_detail.get('uri_slug', '')}/{product_detail.get('product_id', '')}",
                "price": product_detail.get("product_mrp", "N/A"),
                "brand": product_detail.get("brand_name", "N/A"),
                "in_stock": product_detail.get("instock", "").lower() == "yes",
                "stock_status": "" if product_detail.get("instock", "").lower() == "yes" else "Out of Stock",
                "first_image": product_detail.get("product_image", [""])[0] if isinstance(product_detail.get("product_image"), list) else product_detail.get("product_image", ""),
                "features": features,
                "score": 0.0  # API results don't have scores
            })
        
        return processed_products
        
    except Exception as e:
        logger.error(f"API search error: {str(e)}")
        return []

# === Product Response Generation ===
def build_product_prompt(products: List[Dict], query: str, history: str) -> str:
    formatted_products = [
        {
            "name": p.get("name", ""),
            "link": p.get("link", ""),
            "price": p.get("price", ""),
            "image": p.get("first_image", ""),
            "brand": p.get("brand", ""),
            "in_stock": p.get("in_stock", False),
            "features": p.get("features", [])[:4]
        } for p in products
    ]
    
    # Check if query is asking for a specific product
    query_lower = query.lower()
    is_specific_product = any(keyword in query_lower for keyword in [
        'iphone', 'samsung', 'vivo', 'oppo', 'oneplus', 'xiaomi', 'realme',
        'sony', 'lg', 'panasonic', 'daikin', 'voltas', 'carrier', 'hitachi'
    ])
    
    specific_instruction = ""
    if is_specific_product:
        specific_instruction = """
    IMPORTANT: The customer is asking for a specific product. If the exact product is not found:
    - Acknowledge that the specific product wasn't found
    - Show the closest alternatives available
    - Mention that the requested product might not be in stock or available
    - Offer to help find similar products
    """
    
    return f"""
    The customer is ready for product recommendations based on their query and history.

    Chat History:
    {history[-800:]}

    Customer Query:
    {query}

    Available Products:
    {formatted_products}

    Instructions:
    - Output JSON with: "answer" (friendly intro), "products" (max 3-4), "end" (follow-up question)
    - Include only relevant products with name, link, price, image, and 3-4 features (as strings)
    - Skip products missing key fields (price, link)
    - "answer" should be warm and tailored to the query
    - "end" should ask a specific follow-up (e.g., "Do these work for you, or want different options?")
    - If customer mentions selecting a specific option (e.g., "2nd option"), focus on that product
    - If customer wants to buy, provide clear next steps
    - No markdown or extra text, just raw JSON
    {specific_instruction}

    CRITICAL: Respond with ONLY valid JSON. No additional text, no markdown formatting.

    Format:
    {{
        "status": "success",
        "data": {{
            "answer": "Here are some options that match your needs.",
            "products": [
                {{"name": "Product", "link": "url", "price": "₹29999", "image": "url", "features": ["f1", "f2"]}}
            ],
            "end": "Do any of these interest you, or should I refine the search?"
        }}
    }}
    """

async def generate_product_response(products: List[Dict], query: str, history: str) -> str:
    """Generate LLM response for products"""
    if not products:
        return "I couldn't find matching products. Would you like to try different keywords?"
    
    prompt = build_product_prompt(products, query, history)
    response = await generate_with_llm(prompt) or "I'm having trouble generating a response."
    print(f"🔍 LLM Response: {response[:500]}...")  # Show first 500 chars for debugging
    return response

import re
import json
from typing import Dict

def extract_json_from_string(text: str) -> Dict:
    """Strips markdown and extracts clean JSON from a model response"""
    try:
        # Remove ```json and ``` if present
        json_str = re.sub(r"```json|```", "", text).strip()
        
        # Try to find JSON object in the text
        if not json_str.startswith('{'):
            # Look for JSON object in the text
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx + 1]
        
        # print(f"🔍 Attempting to parse JSON: {json_str[:200]}...")
        result = json.loads(json_str)
        print(f"✅ JSON parsed successfully")
        return result
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {str(e)}")
        print(f"🔍 Raw response: {text[:300]}...")
        return {}
    except Exception as e:
        print(f"❌ Unexpected error parsing JSON: {str(e)}")
        print(f"🔍 Raw response: {text[:300]}...")
        return {}

import re

def extract_price_filter(query: str):
    query = query.replace("₹", "").replace(",", "").lower()

    if match := re.search(r"(under|below)\s*(\d+)", query):
        return {"$lte": int(match.group(2))}
    elif match := re.search(r"(above|over)\s*(\d+)", query):
        return {"$gte": int(match.group(2))}
    elif match := re.search(r"between\s*(\d+)\s*and\s*(\d+)", query):
        return {"$gte": int(match.group(1)), "$lte": int(match.group(2))}
    return None

def generate_fallback_response(products: List[Dict], query: str) -> Dict:
    """Generate a fallback response when LLM fails"""
    if not products:
        return {
            "status": "success",
            "data": {
                "answer": "I couldn't find any products matching your search criteria. Could you try searching with different keywords or a different price range?",
                "products": [],
                "intent": "general_search",
                "end": "Is there anything else I can help you with today?"
            }
        }
    
    # Create a simple response with the products
    formatted_products = []
    for product in products[:3]:  # Limit to 3 products
        formatted_products.append({
            "name": product.get("name", ""),
            "link": product.get("link", ""),
            "price": product.get("price", ""),
            "image": product.get("first_image", ""),
            "features": product.get("features", [])[:3]  # Limit to 3 features
        })
    
    # Check for specific product requests
    query_lower = query.lower()
    
    # iPhone specific handling
    if "iphone" in query_lower:
        if "16" in query_lower or "pro max" in query_lower:
            answer = "I couldn't find the exact iPhone 16 Pro Max you're looking for in our current inventory. However, I found some excellent smartphone alternatives that offer similar premium features and performance."
        else:
            answer = "I found some smartphones that offer great value and features, including some premium options."
    
    # Samsung specific handling
    elif "samsung" in query_lower:
        answer = "I found some Samsung smartphones that offer excellent features and performance."
    
    # General smartphone handling
    elif "smartphone" in query_lower or "mobile" in query_lower or "phone" in query_lower:
        answer = "I've found some smartphones that offer great value and features."
    
    # Other specific product categories
    elif "under" in query_lower or "budget" in query_lower:
        answer = "I found some products within your budget that should meet your needs."
    elif "ac" in query_lower or "air conditioner" in query_lower:
        answer = "Here are some AC options from our collection."
    elif "tv" in query_lower or "television" in query_lower:
        answer = "Here are some TVs from our selection."
    elif "laptop" in query_lower:
        answer = "I found some laptops that combine performance and value."
    else:
        answer = "I found some products that match your search."
    
    return {
        "status": "success",
        "data": {
            "answer": answer,
            "products": formatted_products,
            "intent": "general_search",
            "end": "Would you like me to help you with anything else?"
        }
    }

# === Intent Processing ===
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

async def decide_action_with_llm(query: str, history: str) -> Dict:
    """
    Use LLM to decide whether to do RAG conversation or product search.
    Returns a dict with 'action' ('conversation' or 'product_search') and 'reasoning'.
    """
    prompt = f"""
    You are a customer support AI for Lotus Electronics. Analyze the customer's query and decide the best course of action.

    Conversation History:
    {history[-450:]}

    Customer Query: "{query}"

    DECISION TASK:
    Choose between two actions:
    1. "conversation" - Continue chatting, ask questions, provide information, build rapport
    2. "product_search" - Search for and show specific products

    RULES:
    - Choose "conversation" for: greetings, general questions, help requests, vague queries, building rapport
    - Choose "product_search" for: specific product requests, budget mentions, clear shopping intent
    - Consider conversation context and history
    - Prioritize customer experience and natural flow

    Respond in this exact JSON format:
    {{
        "action": "conversation|product_search",
        "reasoning": "Brief explanation of why this action was chosen"
    }}

    Examples:
    - "hello" → {{"action": "conversation", "reasoning": "Greeting requires friendly response"}}
    - "I need a TV under 30000" → {{"action": "product_search", "reasoning": "Specific product request with budget"}}
    - "help me" → {{"action": "conversation", "reasoning": "General help request needs clarification"}}
    - "show me smartphones" → {{"action": "product_search", "reasoning": "Direct product category request"}}
    """

    try:
        response = await generate_with_llm(prompt)
        # print(f"🔍 LLM Decision Response: {response}")
        
        # Parse the JSON response
        parsed = extract_json_from_string(response)
        
        if parsed and "action" in parsed:
            return {
                "action": parsed["action"],
                "reasoning": parsed.get("reasoning", "No reasoning provided"),
                "success": True
            }
        else:
            # Fallback logic if LLM fails
            return fallback_action_decision(query, history)
            
    except Exception as e:
        logger.error(f"LLM decision failed: {str(e)}")
        return fallback_action_decision(query, history)

def fallback_action_decision(query: str, history: str) -> Dict:
    """
    Fallback decision logic when LLM fails.
    """
    query_lower = query.lower()
    
    # Product search keywords
    product_keywords = [
        'show me', 'find', 'search', 'looking for', 'need', 'want to buy',
        'recommend', 'suggest', 'options', 'available', 'price', 'cost',
        'budget', 'under', 'above', 'between', 'around', 'approximately',
        'tv', 'smartphone', 'laptop', 'ac', 'fan', 'refrigerator', 'cooler'
    ]
    
    # Conversation keywords
    conversation_keywords = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'help', 'support', 'question', 'ask', 'tell me', 'explain',
        'how', 'what', 'when', 'where', 'why', 'can you', 'do you'
    ]
    
    # Product selection keywords (should trigger product search to show selected product)
    selection_keywords = [
        'go with', 'choose', 'select', 'option', '1st', '2nd', '3rd', 'first', 'second', 'third',
        'buy', 'purchase', 'order', 'get this', 'want this'
    ]
    
    # Check for selection keywords first (these should show products)
    has_selection = any(keyword in query_lower for keyword in selection_keywords)
    if has_selection:
        return {
            "action": "product_search",
            "reasoning": "Customer is selecting a specific product option",
            "success": True
        }
    
    # Check for product keywords
    has_product = any(keyword in query_lower for keyword in product_keywords)
    has_conversation = any(keyword in query_lower for keyword in conversation_keywords)
    
    if has_product and not has_conversation:
        return {
            "action": "product_search",
            "reasoning": "Query contains product-related keywords",
            "success": True
        }
    elif has_conversation or not has_product:
        return {
            "action": "conversation", 
            "reasoning": "Query appears to be conversational or general",
            "success": True
        }
    else:
        return {
            "action": "conversation",
            "reasoning": "Default to conversation for unclear queries",
            "success": True
        }

async def generate_conversation_response(query: str, history: str) -> str:
    """
    Generate an intelligent conversational response using LLM.
    """
    prompt = f"""
    You are a friendly customer support agent for Lotus Electronics. The customer is engaging in conversation, not asking for specific products yet.

    Conversation History:
    {history[-450:]}

    Customer Query: "{query}"

    TASK: Provide a helpful, friendly response that:
    - Acknowledges their message appropriately
    - Builds rapport and shows you're listening
    - Guides them toward product discovery naturally
    - Asks relevant follow-up questions when appropriate
    - Maintains the conversation flow
    - Handles product selection requests properly (e.g., "go with 2nd option" should refer to the 2nd product from previous results)

    RESPONSE GUIDELINES:
    - Be warm, professional, and helpful
    - Don't immediately jump to product recommendations
    - Ask questions to understand their needs better
    - Provide general information about Lotus Electronics when relevant
    - Keep responses conversational and engaging
    - If customer mentions selecting a specific product option, acknowledge their choice clearly
    - If customer wants to buy, guide them to the product details or purchase process

    Respond with just the conversation text (no JSON, no formatting).
    """

    try:
        response = await generate_with_llm(prompt)
        if response and response.strip():
            return response.strip()
        else:
            # Fallback to simple responses
            return fallback_conversation_response(query)
    except Exception as e:
        logger.error(f"Conversation response generation failed: {str(e)}")
        return fallback_conversation_response(query)

def fallback_conversation_response(query: str) -> str:
    """
    Fallback conversation responses when LLM fails.
    """
    query_lower = query.lower()
    
    # Greeting responses
    if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hello! Welcome to Lotus Electronics. I'm here to help you find the perfect products. What are you looking for today?"
    
    # Help requests
    if 'help' in query_lower:
        return "I'd be happy to help! I can assist you with finding products, checking prices, comparing features, and more. What specific product category are you interested in?"
    
    # General questions
    if any(word in query_lower for word in ['what', 'how', 'tell me', 'explain']):
        return "I can help you with information about our products, prices, features, and availability. What would you like to know more about?"
    
    # Product categories
    if any(category in query_lower for category in ['tv', 'television', 'smartphone', 'mobile', 'phone', 'laptop', 'ac', 'air conditioner', 'fan', 'refrigerator', 'washing machine']):
        return f"Great choice! I can help you find the perfect {query_lower.split()[0]} that fits your needs and budget. What's your budget range for this product?"
    
    # Default response
    return "I'm here to help you find the right products at Lotus Electronics. Could you tell me what type of product you're looking for and your budget?"

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