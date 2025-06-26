import joblib
from sentence_transformers import SentenceTransformer
from typing import Tuple
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
clf = joblib.load(os.path.join(current_dir, 'models', 'intent_classifier.joblib'))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def classify_query(query: str) -> Tuple[str, str]:
    """
    Classify a user query into an intent category.
    
    Args:
        query (str): The user's input query
        
    Returns:
        Tuple[str, str]: A tuple containing (intent, refined_query)
        - intent: The classified intent category
        - refined_query: The original query (can be refined if needed)
    """
    try:
        # Encode the query
        vec = embedding_model.encode([query])
        
        # Predict intent
        intent = clf.predict(vec)[0]
        
        return intent, query
    except Exception as e:
        # Return default intent in case of error
        return "general_search", query

# Intent categories from the trained model
INTENT_CATEGORIES = {
    "general_search",        # For broad product searches
    "specific_product_info", # For specific product details
    "user_specific_query"    # For user account/cart related queries
}

def is_product_query(intent: str) -> bool:
    """
    Check if the intent is related to product queries.
    
    Args:
        intent (str): The classified intent
        
    Returns:
        bool: True if the intent is product-related, False otherwise
    """
    return intent in ["general_search", "specific_product_info"]

if __name__ == '__main__':
    # Test queries
    test_queries = [
        "show me macbook",
        "what phones can I get under 15000",
        "how much does the iPhone 14 Pro cost",
        "show me my cart items",
        "top smartwatches for fitness",
        "cancel my last order",
        "does Galaxy Tab S9 support a stylus",
        "track my delivery",
        "cheap noise cancelling headphones"
    ]
    
    print("\n🧪 Testing Intent Classifier:\n")
    print("-" * 50)
    
    for query in test_queries:
        intent, _ = classify_query(query)
        is_product = is_product_query(intent)
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print(f"Is Product Query: {is_product}")
        print("-" * 50)