
import joblib
from sentence_transformers import SentenceTransformer

# Load once during startup
clf = joblib.load("intent_classifier/models/intent_classifier.joblib")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def classify_query(query: str) -> str:
    """Classifies the intent of a query."""
    vec = embedding_model.encode([query])
    intent = clf.predict(vec)[0]
    return intent
