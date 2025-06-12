import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load training data
df = pd.read_csv('data/training_data.csv', encoding='utf-8')
queries = df['query'].tolist()
labels = df['intent'].tolist()

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
X = embedding_model.encode(queries)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, 'models/intent_classifier.joblib')

print("✅ Training complete. Classifier saved to 'models/intent_classifier.joblib'")
