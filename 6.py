# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 6
# Title: Principal Component Analysis (PCA), step by step
# Dataset: Iris dataset (4 numerical features).
# Shows preprocessing, covariance matrix, eigenvalues/eigenvectors,
# explained variance, transformation and reconstruction.

# 1. AUTOMATIC INSTALLATION
import sys
import subprocess
import importlib.util

packages = {"numpy": "numpy", "matplotlib": "matplotlib",
            "scikit-learn": "sklearn"}

for package, module in packages.items():
    if importlib.util.find_spec(module) is None:
        print("Installing", package)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 2. IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# 3. LOAD DATASET
data = load_iris()
X = data.data
y = data.target
print("Dataset: Iris")
print("Rows:", X.shape[0], "| Original features:", X.shape[1])
print("Feature names:", data.feature_names)
print("First five rows:\n", X[:5])

# 4. STANDARDIZATION (mean approximately 0 and standard deviation approximately 1)
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
print("\nFirst five standardized rows:\n", np.round(X_standard[:5], 3))

# 5. NORMALIZATION (values between 0 and 1; shown as a separate required preprocessing step)
normal_scaler = MinMaxScaler()
X_normalized = normal_scaler.fit_transform(X)
print("\nFirst five normalized rows:\n", np.round(X_normalized[:5], 3))

# 6. CALCULATE COVARIANCE MATRIX FROM STANDARDIZED DATA
cov_matrix = np.cov(X_standard, rowvar=False)
print("\nCovariance Matrix:\n", np.round(cov_matrix, 4))

# 7. CALCULATE EIGENVALUES AND EIGENVECTORS
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort from largest eigenvalue to smallest eigenvalue.
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nEigenvalues:\n", np.round(eigenvalues, 4))
print("\nEigenvectors (columns are principal components):\n",
      np.round(eigenvectors, 4))

# 8. EXPLAINED VARIANCE
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(explained_variance_ratio)
print("\nExplained variance ratio:\n", np.round(explained_variance_ratio, 4))
print("Explained variance (%):\n", np.round(explained_variance_ratio * 100, 2))
print("Cumulative variance (%):\n", np.round(cumulative_variance * 100, 2))
print("Variance retained by first two components (%):",
      round(cumulative_variance[1] * 100, 2))

# 9. TRANSFORM DATA: project 4 features onto the first 2 principal components.
components = eigenvectors[:, :2]
X_reduced = X_standard @ components
print("\nReduced data shape:", X_reduced.shape)  # expected: 150 rows x 2 columns

# 10. RECONSTRUCT DATA from the two selected components.
X_reconstructed = X_reduced @ components.T
reconstruction_mse = np.mean((X_standard - X_reconstructed) ** 2)
print("Reconstructed data shape:", X_reconstructed.shape)
print("Reconstruction MSE (standardized feature space):",
      round(reconstruction_mse, 5))

# 11. VERIFY USING SCIKIT-LEARN PCA
pca = PCA(n_components=2)
X_sklearn = pca.fit_transform(X_standard)
print("\nScikit-learn PCA explained variance ratio:",
      np.round(pca.explained_variance_ratio_, 4))

# 12. VISUALIZE ORIGINAL FEATURES AND REDUCED DATA
plt.figure(figsize=(7, 5))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset after Manual PCA")
plt.tight_layout()
plt.show()

# Plot explained variance for each component.
plt.figure(figsize=(7, 4))
plt.bar(["PC1", "PC2", "PC3", "PC4"], explained_variance_ratio * 100)
plt.ylabel("Explained variance (%)")
plt.title("PCA Explained Variance")
plt.tight_layout()
plt.show()

# # EXPERIMENT 6: PCA, SVD AND LDA

# import sys
# import subprocess
# import importlib.util

# # AUTOMATIC INSTALLATION
# packages = {
#     "numpy": "numpy",
#     "matplotlib": "matplotlib",
#     "scikit-learn": "sklearn"
# }

# for package, module in packages.items():
#     if importlib.util.find_spec(module) is None:
#         subprocess.check_call(
#             [sys.executable, "-m", "pip", "install", package]
#         )

# # IMPORT LIBRARIES
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# # LOAD DATASET
# data = load_iris()
# X = data.data
# y = data.target

# # STANDARDIZE DATA
# X_scaled = StandardScaler().fit_transform(X)

# # PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# print("PCA OUTPUT SHAPE:", X_pca.shape)
# print("PCA EXPLAINED VARIANCE:", pca.explained_variance_ratio_)

# # SVD
# svd = TruncatedSVD(n_components=2, random_state=42)
# X_svd = svd.fit_transform(X_scaled)

# print("\nSVD OUTPUT SHAPE:", X_svd.shape)
# print("SVD EXPLAINED VARIANCE:", svd.explained_variance_ratio_)

# # LDA
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_lda = lda.fit_transform(X_scaled, y)

# print("\nLDA OUTPUT SHAPE:", X_lda.shape)

# # VISUALIZATION
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y)
# axes[0].set_title("PCA")

# axes[1].scatter(X_svd[:, 0], X_svd[:, 1], c=y)
# axes[1].set_title("SVD")

# axes[2].scatter(X_lda[:, 0], X_lda[:, 1], c=y)
# axes[2].set_title("LDA")

# plt.tight_layout()
# plt.show()