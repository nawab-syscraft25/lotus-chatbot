import joblib
from sentence_transformers import SentenceTransformer

# Load models
clf = joblib.load('models/intent_classifier.joblib')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load this normally

# Example test queries
test_queries = [
    "what phones can I get under 15000",
    "how much does the iPhone 14 Pro cost",
    "show me my cart items",
    "top smartwatches for fitness",
    "cancel my last order",
    "does Galaxy Tab S9 support a stylus",
    "track my delivery",
    "cheap noise cancelling headphones"
]

print("🧪 Running Test Cases:\n")

for query in test_queries:
    vec = embedding_model.encode([query])
    intent = clf.predict(vec)[0]
    print(f"Query: {query}")
    print(f"Predicted Intent: {intent}\n")
