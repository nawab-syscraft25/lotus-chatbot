import requests
import json
from typing import Dict, Optional, Tuple
from functools import lru_cache
import re
from urllib.parse import urlparse

# Constants
API_URL = "https://portal.lotuselectronics.com/web-api/home/product_detail"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "auth-token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNjg5MzYiLCJpYXQiOjE3NDg5NDc2NDEsImV4cCI6MTc0ODk2NTY0MX0.uZeQseqc6mpm5vkOAmEDgUeWIfOI5i_FnHJRaUBWlMY",
    "content-type": "application/x-www-form-urlencoded",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0"
}
PRODUCT_ID_PATTERN = re.compile(r"/(\d+)/?$")
REQUEST_TIMEOUT = 10  # seconds

def extract_product_id_from_url(url: str) -> Optional[str]:
    """Extract product ID from URL using regex for better reliability."""
    try:
        match = PRODUCT_ID_PATTERN.search(url)
        return match.group(1) if match else None
    except Exception:
        return None

@lru_cache(maxsize=1024)
def get_product_details(product_id: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check product stock status with caching.
    Returns tuple of (is_in_stock, product_details)
    """
    try:
        data = {
            "product_id": product_id,
            "cat_name": f"/product/{product_id}",
            "product_name": f"product-{product_id}"
        }

        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("data", {}).get("product_detail"):
            return False, None

        detail = result["data"]["product_detail"]
        instock = detail.get("instock", "").lower()
        out_of_stock = detail.get("out_of_stock", "0")
        quantity = int(detail.get("product_quantity", "0"))

        is_in_stock = (
            instock == "yes" and
            out_of_stock == "0" and
            quantity > 0
        )

        return is_in_stock, detail

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return False, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False, None

def get_first_image(product_detail: Dict) -> Optional[str]:
    """Extract first available image URL from product details."""
    for field in ["product_image", "product_images_350"]:
        if product_detail.get(field):
            field_value = product_detail[field]
            if isinstance(field_value, list) and len(field_value) > 0:
                return field_value[0]
            elif isinstance(field_value, str):
                return field_value
    return None

def get_product_stock_status(product_link: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Optimized product stock check with better error handling.
    Returns tuple of (is_in_stock, first_image_url, error_message)
    """
    try:
        product_id = extract_product_id_from_url(product_link)
        if not product_id:
            return False, None, "Invalid product URL format"

        is_in_stock, product_details = get_product_details(product_id)
        
        if not product_details:
            return False, None, "Product details not found"

        return is_in_stock, get_first_image(product_details), None

    except Exception as e:
        return False, None, f"System error: {str(e)}"

# Example usage with improved output
if __name__ == "__main__":
    test_urls = [
        "https://www.lotuselectronics.com/product/qled-tv/haier-qled-tv-165-cm-65-inches-android-65s800qt-black/38455",
        "https://www.lotuselectronics.com/product/invalid-url",
        "https://www.lotuselectronics.com/product/smartphones/samsung-galaxy-s21/12345"
    ]

    for url in test_urls:
        print(f"\nChecking: {url}")
        in_stock, image, error = get_product_stock_status(url)
        
        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"✅ In Stock: {in_stock}")
            print(f"🖼️ First Image: {image or 'Not available'}")