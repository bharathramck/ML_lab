import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score

data = pd.read_csv("Heart_Disease.csv")
print(data.head())

# Encode categorical columns if any
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

k = 3

# KMeans Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
kmeans_labels = kmeans.labels_

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=k, random_state=42)
gmm.fit(scaled_data)
gmm_labels = gmm.predict(scaled_data)

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=gmm_labels, cmap='viridis')
plt.title('EM (GMM) Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Silhouette Score for KMeans and GMM
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
gmm_silhouette = silhouette_score(scaled_data, gmm_labels)

if 'Heart Disease' in data.columns:
    true_labels = data['Heart Disease']
    ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
    ari_gmm = adjusted_rand_score(true_labels, gmm_labels)
    print(f'K-Means ARI: {ari_kmeans}')
    print(f'GMM ARI: {ari_gmm}')

print(f'K-Means Silhouette Score: {kmeans_silhouette}')
print(f'GMM Silhouette Score: {gmm_silhouette}')
