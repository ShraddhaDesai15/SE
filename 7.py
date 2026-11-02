# EXPERIMENT 7: GRAPH-BASED CLUSTERING

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
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

# GENERATE SAMPLE DATA
X, actual_labels = make_moons(
    n_samples=300, noise=0.08, random_state=42
)

# SPECTRAL CLUSTERING
spectral = SpectralClustering(
    n_clusters=2,
    affinity="nearest_neighbors",
    n_neighbors=10,
    random_state=42
)

spectral_labels = spectral.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# DISPLAY RESULTS
print("SPECTRAL CLUSTER LABELS:", set(spectral_labels))
print("DBSCAN CLUSTER LABELS:", set(dbscan_labels))
print("DBSCAN NOISE POINTS:", sum(dbscan_labels == -1))

if len(set(spectral_labels)) > 1:
    print(
        "SPECTRAL SILHOUETTE SCORE:",
        silhouette_score(X, spectral_labels)
    )

# VISUALIZATION
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=actual_labels)
axes[0].set_title("Original Data")

axes[1].scatter(X[:, 0], X[:, 1], c=spectral_labels)
axes[1].set_title("Spectral Clustering")

axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels)
axes[2].set_title("DBSCAN")

plt.tight_layout()
plt.show()