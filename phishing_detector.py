import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data["EmailText"]
y = data["Label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n=== PHISHING EMAIL DETECTION MODEL ===\n")

print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Confusion Matrix
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test custom email
print("\n=== CUSTOM EMAIL TEST ===\n")

custom_email = input("Enter an email message: ")

custom_vector = vectorizer.transform([custom_email])

prediction = model.predict(custom_vector)

if prediction[0] == "Phishing":
    print("\nResult: PHISHING EMAIL")
else:
    print("\nResult: SAFE EMAIL")