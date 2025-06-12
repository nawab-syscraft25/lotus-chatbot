import requests
import json
from typing import Dict, Optional, Tuple

def extract_product_id_from_url(url: str) -> Optional[str]:
    """Extract product ID from Lotus Electronics product URL."""
    try:
        # Split by / and get the last part which should contain the product ID
        parts = url.strip('/').split('/')
        if len(parts) >= 2:
            return parts[-1]
        return None
    except Exception:
        return None

def get_product_details(product_id: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check product stock status and get details from Lotus Electronics API.
    Returns a tuple of (is_in_stock, product_details)
    """
    try:
        url = "https://portal.lotuselectronics.com/web-api/home/product_detail"
        
        headers = {
            "accept": "application/json, text/plain, */*",
            "auth-key": "Web2@!9",
            "auth-token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNjg5MzYiLCJpYXQiOjE3NDg5NDc2NDEsImV4cCI6MTc0ODk2NTY0MX0.uZeQseqc6mpm5vkOAmEDgUeWIfOI5i_FnHJRaUBWlMY",
            "content-type": "application/x-www-form-urlencoded",
            "end-client": "Lotus-Web",
            "origin": "https://www.lotuselectronics.com",
            "referer": "https://www.lotuselectronics.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0"
        }

        # Extract product name and category from URL
        product_name = f"product-{product_id}"  # This is a placeholder, you might want to extract actual name
        cat_name = f"/product/{product_id}"  # This is a placeholder, you might want to extract actual category

        data = {
            "product_id": product_id,
            "cat_name": cat_name,
            "product_name": product_name
        }

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        
        result = response.json()
        
        if "data" in result and "product_detail" in result["data"]:
            product_detail = result["data"]["product_detail"]
            
            # Check stock status from multiple fields
            instock = product_detail.get("instock", "").lower()
            out_of_stock = product_detail.get("out_of_stock", "0")
            product_quantity = int(product_detail.get("product_quantity", "0"))
            
            # Product is in stock only if:
            # 1. instock is explicitly "yes"
            # 2. out_of_stock is "0"
            # 3. product_quantity is greater than 0
            is_in_stock = (
                instock == "yes" and
                out_of_stock == "0" and
                product_quantity > 0
            )
            
            # Get first image URL
            first_image = None
            if "product_image" in product_detail and product_detail["product_image"]:
                first_image = product_detail["product_image"][0]
            elif "product_images_350" in product_detail and product_detail["product_images_350"]:
                first_image = product_detail["product_images_350"][0]
            
            return is_in_stock, product_detail
            
        return False, None

    except Exception as e:
        print(f"Error checking product status: {str(e)}")
        return False, None

def get_product_stock_status(product_link: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if a product is in stock and get its first image.
    Returns a tuple of (is_in_stock, first_image_url, error_message)
    """
    try:
        product_id = extract_product_id_from_url(product_link)
        if not product_id:
            return False, None, "Invalid product URL"

        is_in_stock, product_details = get_product_details(product_id)
        
        if product_details:
            # Get the first image URL if available
            first_image = None
            if "product_image" in product_details and product_details["product_image"]:
                first_image = product_details["product_image"][0]
            elif "product_images_350" in product_details and product_details["product_images_350"]:
                first_image = product_details["product_images_350"][0]
            
            return is_in_stock, first_image, None
        
        return False, None, "Product details not found"

    except Exception as e:
        return False, None, f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    test_url = "https://www.lotuselectronics.com/product/qled-tv/haier-qled-tv-165-cm-65-inches-android-65s800qt-black/38455"
    is_in_stock, first_image, error = get_product_stock_status(test_url)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Product in stock: {is_in_stock}")
        print(f"First image URL: {first_image}") 