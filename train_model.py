import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load downloaded dataset
data = pd.read_parquet(
    "train-00000-of-00001-290868f0a36350c5.parquet"
)

print("Dataset columns:", data.columns.tolist())

# Check and clean data
data = data.dropna(subset=["text", "label"])
data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(str).str.upper().str.strip()

# Features and labels
X = data["text"]
y = data["label"]

print("Total news:", len(data))
print("Label counts:")
print(y.value_counts())
print("Label 0 sample:", data[data["label"] == "0"]["text"].iloc[0])
print("Label 1 sample:", data[data["label"] == "1"]["text"].iloc[0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Evaluate model
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, predictions))

# Save model
with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")
