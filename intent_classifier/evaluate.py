import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report

# Load test data
df = pd.read_csv("data/test_data.csv")
queries = df["query"].tolist()
true_labels = df["intent"].tolist()

# Load models
clf = joblib.load("models/intent_classifier.joblib")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode and predict
X_test = embedding_model.encode(queries)
predicted_labels = clf.predict(X_test)

# Report
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels)

print(f"✅ Accuracy: {accuracy:.2f}\n")
print("📊 Classification Report:\n")
print(report)
