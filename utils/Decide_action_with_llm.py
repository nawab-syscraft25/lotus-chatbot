import logging
from typing import Dict, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import httpx  
from utils.generate_llm import generate_with_llm

from utils.Search_lotus_products import  extract_json_from_string


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