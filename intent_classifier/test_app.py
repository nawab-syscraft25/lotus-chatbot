from fastapi import FastAPI, Form
import joblib
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

clf = joblib.load('models/intent_classifier.joblib')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_product_name(query: str) -> str:
    query = query.lower()
    keywords_to_remove = [
        "i want to buy", "price of", "what is the price of",
        "tell me about", "show me", "how much is"
    ]
    for phrase in keywords_to_remove:
        if phrase in query:
            query = query.replace(phrase, "")
    return query.strip()

@app.post("/chat")
def handle_query(query: str = Form(...)):
    vec = embedding_model.encode([query])
    intent = clf.predict(vec)[0]

    if intent == "specific_product_info":
        product_name = extract_product_name(query)

        try:
            response = requests.post(
                url="https://portal.lotuselectronics.com/web-api/home/search_suggestion",
                headers={
                    "auth-key": "Web2@!9",
                    "end-client": "Lotus-Web"
                },
                data={"search_text": product_name}
            )
            response.raise_for_status()
            results = response.json()
            return {"intent": intent, "query": query, "search_text": product_name, "result": results}
        except requests.exceptions.RequestException as e:
            return {"intent": intent, "error": str(e)}

    return {"intent": intent, "message": "No product info API call needed."}
