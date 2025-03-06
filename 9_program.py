import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nCorrect Predictions:")
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"Predicted: {iris.target_names[y_pred[i]]}, Actual: {iris.target_names[y_test[i]]}")

print("\nWrong Predictions:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print(f"Predicted: {iris.target_names[y_pred[i]]}, Actual: {iris.target_names[y_test[i]]}")
