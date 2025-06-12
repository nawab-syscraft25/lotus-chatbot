from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import uuid
import asyncio
import signal
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Dict, List
import uvicorn

# === Modular Helpers ===
from utils.product_image_extractor import ProductImageExtractor
from intent_classifier import classify_query
from search_product import search_products
from vector_search import search_vector_db

# === Load Environment ===
load_dotenv()

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

# === URL Normalization Middleware ===
@app.middleware("http")
async def normalize_url(request: Request, call_next):
    # Normalize the URL path by removing double slashes
    path = request.url.path.replace("//", "/")
    if path != request.url.path:
        # Create new URL with normalized path
        url = request.url.replace(path=path)
        # Create new request with normalized URL
        request._url = url
    response = await call_next(request)
    return response

# === Static Files and Templates ===
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

API_KEY = "nawabkhan"
BUFFER_MEMORY_SIZE = 5

# === In-Memory Chat Memory ===
class ChatMemory:
    def __init__(self):
        self.memory: Dict[str, List[Dict]] = {}

    def get_memory(self, session_id: str) -> List[Dict]:
        return self.memory.get(session_id, [])

    def add_to_memory(self, session_id: str, message: Dict):
        self.memory.setdefault(session_id, []).append(message)

    def clear_memory(self, session_id: str):
        self.memory.pop(session_id, None)

chat_memory = ChatMemory()
product_extractor = ProductImageExtractor()

# === Models ===
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

# === Auth Middleware ===
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# === Startup / Shutdown ===
@app.on_event("startup")
async def startup_event():
    print("Starting Lotus Shopping Assistant...")

@app.on_event("shutdown")
async def shutdown_event():
    global is_shutting_down
    is_shutting_down = True
    print("Shutting down Lotus Shopping Assistant...")
    await asyncio.sleep(1)

# === Signal Handling ===
def handle_sigterm(signum, frame):
    print("Received SIGTERM. Shutting down...")
    asyncio.create_task(shutdown_event())

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# === Routes ===
@app.get("/")
async def root(request: Request, accept: str = Header(None)):
    if accept and "application/json" in accept:
        return JSONResponse({
            "status": "success",
            "data": {
                "service": "Lotus Shopping Assistant",
                "version": "1.0.0",
                "endpoints": {
                    "chat": "/ask",
                    "products": "/api/product",
                    "search": "/api/product/search"
                }
            }
        })
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "status": "success" if not is_shutting_down else "error",
        "data": {
            "service": "Lotus Shopping Assistant",
            "version": "1.0.0",
            "status": "operational" if not is_shutting_down else "shutting_down"
        }
    })

@app.get("/api/product/{product_id}")
async def get_product(product_id: str, api_key: str = Depends(verify_api_key)):
    return {"product_id": product_id, "name": "Sample Product"}

@app.get("/api/product/search")
async def search_products(query: str, api_key: str = Depends(verify_api_key)):
    return {"results": [{"id": 1, "name": "Sample Product"}]}

@app.post("/ask")
async def ask(question: Question, api_key: str = Depends(verify_api_key)):
    if is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Server is shutting down"}
        )
        
    try:
        session_id = question.session_id or str(uuid.uuid4())
        history = chat_memory.get_memory(session_id)
        chat_history_str = "\n".join(
            [f"User: {msg['user']}\nBot: {msg['bot']}" for msg in history[-BUFFER_MEMORY_SIZE:]]
        )

        # Intent classification
        intent = classify_query(question.question)

        # Refine query
        refined_query = refine_query_with_llm(question.question, chat_history_str)

        # Generate RAG-based answer
        answer = rag_answer(refined_query, chat_history_str, question.question)

        # Save chat
        chat_memory.add_to_memory(session_id, {
            "user": question.question,
            "bot": answer
        })

        return JSONResponse({
            "status": "success",
            "data": {
                "answer": answer,
                "session_id": session_id,
                "intent": intent
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# === RAG Functions ===
def refine_query_with_llm(user_query, chat_history):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    prompt = (
        "You are a smart assistant. Rewrite the user's question to be clear and precise for product search, "
        "using INR (₹) for any prices. Use the conversation context below:\n"
        f"Chat:\n{chat_history}\n"
        f"User: {user_query}\n"
        "Refined query:"
    )
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else user_query

def rag_answer(refined_query, chat_history, original_question):
    refined_query = refined_query.replace('$', '₹').replace('INR', '₹').replace('rupees', '₹').replace('Rs.', '₹').replace('Rs', '₹')
    if '₹' not in refined_query:
        refined_query = refined_query.replace('under ', 'under ₹')

    results = search_vector_db(refined_query)
    docs = results.get("results", [])  # Already limited to 3 and sorted
    if not docs:
        return "We're sorry, but we couldn't find relevant products. Would you like to try different keywords?"

    context = "\n\n".join(
        (
            f"{doc.get('name', 'Product')}\n"
            f"Link: {doc.get('link', '#')}\n"
            f"Price: ₹{doc.get('price', 'N/A')}\n"
            + (f"Model: {doc.get('brand', '')}\n" if doc.get('brand') else "")
            + (f"Stock Status: {'✅ In Stock' if doc.get('in_stock') else '❌ Out of Stock'}\n")
            + (f"Note: {doc.get('stock_message', '')}\n" if not doc.get('in_stock') else "")
            + (f"Product Image: {doc.get('first_image', 'Image not available')}\n" if doc.get('first_image') else "")
            + (f"Features:\n" + "\n".join(f"- {f}" for f in doc.get('features', [])) + "\n" if doc.get('features') else "")
            + "---"
        )
        for doc in docs
    )

    # Determine product type and generate appropriate follow-up questions
    product_type = "smartphone" if any(word in refined_query.lower() for word in ["phone", "smartphone", "mobile"]) else \
                  "television" if any(word in refined_query.lower() for word in ["tv", "television", "smart tv"]) else \
                  "laptop" if any(word in refined_query.lower() for word in ["laptop", "notebook", "computer"]) else \
                  "product"

    follow_up_questions = {
        "smartphone": "To help you better, we'd like to know:\n1. What's your preferred screen size? (Small: 5-6\", Medium: 6-6.5\", Large: 6.5\"+)\n2. How important is camera quality? (Basic, Good, Professional)\n3. What's your priority? (Battery Life, Performance, Camera, Gaming)",
        "television": "To help you better, we'd like to know:\n1. What's your preferred screen size? (Small: 32-43\", Medium: 43-55\", Large: 55\"+)\n2. What's your viewing distance? (Close: 5-8ft, Medium: 8-12ft, Far: 12ft+)\n3. What's your priority? (Picture Quality, Smart Features, Sound Quality, Gaming)",
        "laptop": "To help you better, we'd like to know:\n1. What's your primary use? (Work, Gaming, Creative Work, Basic Use)\n2. How important is portability? (Very, Somewhat, Not Important)\n3. What's your priority? (Performance, Battery Life, Display Quality, Portability)",
        "product": "To help you better, we'd like to know:\n1. What's your budget range?\n2. What features are most important to you?\n3. Do you have any specific brand preferences?"
    }

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    prompt = (
        "You are a friendly and knowledgeable product advisor for Lotus Electronics. Use chat context and product info below and you are able to complair the products also:\n\n"
        f"Chat:\n{chat_history}\n\n"
        f"Product Info:\n{context}\n\n"
        f"User Question: {original_question}\n\n"
        "Instructions:\n"
        "- Present yourself as part of the Lotus Electronics team using 'we' instead of 'I' or not use I'm\n"
        "- Format each product EXACTLY like this example:\n"
        "---\n"
        "Product Name\n"
        "Link: [direct product link]\n"
        "Price: ₹[price]\n"
        "Model: [model_number]\n"
        "Stock Status: [✅ In Stock or ❌ Out of Stock]\n"
        "Note: [Show stock message for out-of-stock items]\n"
        "Product Image: [image_url]\n"
        "Features:\n"
        "- [feature 1]\n"
        "- [feature 2]\n"
        "- [feature 3]\n"
        "- [feature 4]\n"
        "- [feature 5]\n"
        "- [feature 6]\n"
        "---\n"
        "- Be warm and friendly in your tone\n"
        "- Include the following follow-up question:\n"
        f"{follow_up_questions[product_type]}\n"
        "- Skip missing fields, and never show 'undefined'\n"
        "- Try to Show at least 3 to 4 features\n"
        "- IMPORTANT: Follow the exact formatting shown in the example above\n"
        "- IMPORTANT: Use plain text formatting (no markdown or HTML)"
    )
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else "We apologize, but we encountered a response formatting error. Please try again."

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
