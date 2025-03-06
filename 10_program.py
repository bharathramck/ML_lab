import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_regression(X, y, query_point, tau):
  m = X.shape[0]
  weights = np.exp(-np.sum((X - query_point) ** 2, axis=1) / (2 * tau**2))
  W = np.diag(weights)

  X_b = np.hstack((np.ones((m, 1)), X))
  theta = np.linalg.inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
  return theta

def predict(X, y, query_points, tau):
  predictions = []
  for query_point in query_points:
    theta = locally_weighted_regression(X, y, query_point, tau)
    prediction = np.dot(np.array([1, query_point]), theta)
    predictions.append(prediction)
  return np.array(predictions)

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

query_points = np.linspace(0, 5, 100)

tau = 0.5
predictions = predict(X, y, query_points, tau)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(query_points, predictions, color='red', label='Locally Weighted Regression', linewidth=2)
plt.title('Locally Weighted Regression (LWR)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
