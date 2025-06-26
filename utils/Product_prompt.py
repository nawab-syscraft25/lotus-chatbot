# utils/Product_prompt.py
import asyncio
from typing import Dict, List
import httpx  
from utils.generate_llm import generate_with_llm, run_in_threadpool



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
    {chat_history[-450:]}

    Customer Query:
    {user_query}

    Optimized Search Query:"""

    refined = await generate_with_llm(prompt)
    print(f"🔍 Query refinement: '{user_query}' -> '{refined}'")
    return refined if refined else user_query




