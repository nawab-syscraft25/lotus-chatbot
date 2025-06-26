import os
import httpx
import logging
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.lotuselectronics.com/admin/v6/"
CREATE_ORDER_URL = BASE_URL + "order/create_order"
ADD_TO_CART_URL = BASE_URL + "user/add_products_on_user_cart"
GET_CART_URL = BASE_URL + "user/get_user_cart_data"
GET_ADDRESS_URL = BASE_URL + "user/get_address_list"
SET_DEFAULT_ADDRESS_URL = BASE_URL + "user/set_default_address"

logger = logging.getLogger(__name__)

# === Helper to make POST request ===
async def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            return res.json()
    except Exception as e:
        logger.error(f"HTTP error calling {url}: {e}")
        return {"status": "error", "message": str(e)}

# === Step 1: Add product to cart ===
async def add_product_to_cart(user_id: str, product_id: str, quantity: int = 1):
    payload = {
        "user_id": user_id,
        "products": [
            {
                "product_id": product_id,
                "qty": quantity
            }
        ]
    }
    return await post_json(ADD_TO_CART_URL, payload)

# === Step 2: Get default address or ask user to set one ===
async def get_default_address_id(user_id: str) -> str:
    payload = {"user_id": user_id}
    resp = await post_json(GET_ADDRESS_URL, payload)

    addresses = resp.get("data", [])
    for addr in addresses:
        if addr.get("is_default") == "1":
            return addr.get("address_id")
    if addresses:
        return addresses[0].get("address_id")
    return ""

# === Step 3: Create order ===
async def create_order(user_id: str, address_id: str) -> Dict[str, Any]:
    payload = {
        "user_id": user_id,
        "delivery_address_id": address_id
    }
    return await post_json(CREATE_ORDER_URL, payload)

# === Main Order Handler ===
async def handle_order_placement(user_input: str, session_id: str) -> Dict[str, Any]:
    # This function assumes you've already captured product_id and user_id from prior auth flow
    user_state = get_user_state(session_id)
    user_id = user_state.get("user_id")
    product_id = user_state.get("pending_product_id")

    if not user_id or not product_id:
        return {"status": "error", "message": "User not authenticated or product not selected."}

    # Step 1: Add to cart
    cart_resp = await add_product_to_cart(user_id, product_id)
    if cart_resp.get("status") != "success":
        return {"status": "error", "message": "Failed to add product to cart."}

    # Step 2: Get address
    address_id = await get_default_address_id(user_id)
    if not address_id:
        return {
            "status": "error",
            "message": "No address found. Please add an address in your profile before placing an order."
        }

    # Step 3: Create order
    order_resp = await create_order(user_id, address_id)
    if order_resp.get("status") == "success":
        return {
            "status": "success",
            "data": {
                "answer": "Your order has been successfully placed ✅",
                "order_id": order_resp.get("order_id", ""),
                "intent": "order_confirmation"
            }
        }

    return {
        "status": "error",
        "message": order_resp.get("message", "Failed to place the order.")
    }

# === Dummy user session state manager ===
# Replace with Redis, DB, or memory state logic
_session_memory: Dict[str, Dict] = {}

def get_user_state(session_id: str) -> Dict:
    return _session_memory.get(session_id, {})

def set_user_state(session_id: str, data: Dict):
    _session_memory[session_id] = {**_session_memory.get(session_id, {}), **data}
