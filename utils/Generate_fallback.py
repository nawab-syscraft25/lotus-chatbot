import httpx  
from typing import Optional, Dict, List


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