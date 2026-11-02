import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_parquet(
    "train-00000-of-00001-290868f0a36350c5.parquet"
)

# Remove missing values
data = data.dropna(subset=["text", "label"])

# Use title + article text
data["title"] = data["title"].fillna("")
data["text"] = data["text"].astype(str)
data["title"] = data["title"].astype(str)

data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"].astype(str).str.strip()
print("\nLABEL COUNTS:")
print(data["label"].value_counts())

print("\nSAMPLE NEWS FOR EACH LABEL:")

for label in data["label"].unique():
    print("\nLABEL:", label)

    samples = data[data["label"] == label].head(3)

    for _, row in samples.iterrows():
        print("TITLE:", row["title"])
        print("TEXT:", str(row["text"])[:300])

print("Total news:", len(data))
print("\nLabel counts:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Memory-light TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

print("\nCreating TF-IDF features...")

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("TF-IDF features created!")
print("Features:", X_train_vectorized.shape)

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining model...")
model.fit(X_train_vectorized, y_train)

# Test
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL TRAINED SUCCESSFULLY")
print("==============================")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel saved successfully!")
print("Vectorizer saved successfully!")
print("\nTraining complete!")