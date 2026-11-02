# EXPERIMENT 6: PCA, SVD AND LDA

import sys
import subprocess
import importlib.util

# AUTOMATIC INSTALLATION
packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn"
}

for package, module in packages.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )

# IMPORT LIBRARIES
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LOAD DATASET
data = load_iris()
X = data.data
y = data.target

# STANDARDIZE DATA
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA OUTPUT SHAPE:", X_pca.shape)
print("PCA EXPLAINED VARIANCE:", pca.explained_variance_ratio_)

# SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_scaled)

print("\nSVD OUTPUT SHAPE:", X_svd.shape)
print("SVD EXPLAINED VARIANCE:", svd.explained_variance_ratio_)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

print("\nLDA OUTPUT SHAPE:", X_lda.shape)

# VISUALIZATION
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y)
axes[0].set_title("PCA")

axes[1].scatter(X_svd[:, 0], X_svd[:, 1], c=y)
axes[1].set_title("SVD")

axes[2].scatter(X_lda[:, 0], X_lda[:, 1], c=y)
axes[2].set_title("LDA")

plt.tight_layout()
plt.show()