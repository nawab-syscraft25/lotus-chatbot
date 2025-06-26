import os
import requests
import logging

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LOTUS_API_BASE", "https://www.lotuselectronics.com/admin/v6/")
SEND_OTP_URL = BASE_URL + "user/send_otp"
VERIFY_OTP_URL = BASE_URL + "user/signup_verify"
CHECK_USER_URL = BASE_URL + "user/check_user"
SIGNUP_URL = BASE_URL + "user/signup"
LOGIN_URL = BASE_URL + "user/signin"

logger = logging.getLogger(__name__)
user_sessions = {}  # session_id -> {phone, otp_verified, user_id}

def send_otp(phone: str) -> bool:
    try:
        res = requests.post(SEND_OTP_URL, json={"mobile": phone})
        return res.status_code == 200 and res.json().get("status")
    except Exception as e:
        logger.error(f"Failed to send OTP: {e}")
        return False

def verify_otp(phone: str, otp: str) -> dict:
    try:
        res = requests.post(VERIFY_OTP_URL, json={"mobile": phone, "otp": otp})
        return res.json() if res.status_code == 200 else {}
    except Exception as e:
        logger.error(f"OTP verification failed: {e}")
        return {}

def check_user_exists(phone: str) -> bool:
    try:
        res = requests.post(CHECK_USER_URL, json={"mobile": phone})
        return res.status_code == 200 and res.json().get("is_exist")
    except Exception as e:
        logger.error(f"User check failed: {e}")
        return False

def create_user(phone: str) -> bool:
    try:
        res = requests.post(SIGNUP_URL, json={"mobile": phone})
        return res.status_code == 200 and res.json().get("status")
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        return False

def login_user(phone: str) -> dict:
    try:
        res = requests.post(LOGIN_URL, json={"mobile": phone})
        return res.json() if res.status_code == 200 else {}
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return {}

async def handle_user_authentication(question: str, session_id: str):
    session = user_sessions.get(session_id, {})

    if not session.get("phone"):
        phone = extract_phone_from_question(question)
        if not phone:
            return {"status": "auth", "data": {"prompt": "Please share your registered phone number."}}

        user_sessions[session_id] = {"phone": phone, "otp_verified": False}
        if not send_otp(phone):
            return {"status": "error", "message": "Unable to send OTP, try again."}

        return {"status": "auth", "data": {"prompt": f"OTP sent to {phone}. Please enter the OTP."}}

    if not session.get("otp_verified"):
        otp = extract_otp_from_question(question)
        if not otp:
            return {"status": "auth", "data": {"prompt": "Enter the 4-digit OTP sent to your number."}}

        verify_res = verify_otp(session["phone"], otp)
        if not verify_res.get("status"):
            return {"status": "auth", "data": {"prompt": "Invalid OTP. Please try again."}}

        # Check or create user
        if not check_user_exists(session["phone"]):
            if not create_user(session["phone"]):
                return {"status": "error", "message": "Failed to create account."}

        login_res = login_user(session["phone"])
        if not login_res.get("status"):
            return {"status": "error", "message": "Login failed. Please try again."}

        session.update({"otp_verified": True, "user_id": login_res.get("user_id")})
        return {"status": "success", "data": {"message": "✅ You are now logged in."}}

    return {"status": "success", "data": {"message": "You are already logged in."}}

def extract_phone_from_question(text: str) -> str:
    import re
    match = re.search(r"\b(\d{10})\b", text)
    return match.group(1) if match else ""

def extract_otp_from_question(text: str) -> str:
    import re
    match = re.search(r"\b(\d{4,6})\b", text)
    return match.group(1) if match else ""
