import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the dataset (Ensure it has enough data per class)
data = pd.read_csv('your_dataset.csv')

print("Sample Data:")
print(data.head())

# Features and labels
X = data['text']
y = data['label']

# Check class distribution
print("\nClass distribution:\n", y.value_counts())

# Use stratified split to keep class balance in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Predict on test set
y_pred = classifier.predict(X_test_counts)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

# Print metrics
print(f'\nAccuracy : {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall   : {recall:.2f}')
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
