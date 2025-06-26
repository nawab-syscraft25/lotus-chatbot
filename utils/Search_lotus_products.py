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

import json
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_KEY = "nawabkhan"
BUFFER_MEMORY_SIZE = 5
LLM_TIMEOUT = 8
MAX_WORKERS = 8
PRODUCT_PROCESS_LIMIT = 3 
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

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

async_client = httpx.AsyncClient(timeout=10.0)


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