# EXPERIMENT 5: SUPPORT VECTOR MACHINE

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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# LOAD DATASET
data = load_breast_cancer()
X = data.data
y = data.target

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# BASIC SVM MODEL
model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("BASIC SVM ACCURACY:", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))

# PARAMETER TUNING
parameters = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"]
}

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    parameters,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
tuned_pred = best_model.predict(X_test)

print("\nBEST PARAMETERS:", grid.best_params_)
print("TUNED SVM ACCURACY:", accuracy_score(y_test, tuned_pred))
print("TUNED CONFUSION MATRIX:\n", confusion_matrix(y_test, tuned_pred))

# CONFUSION MATRIX GRAPH
ConfusionMatrixDisplay.from_predictions(
    y_test, tuned_pred, display_labels=data.target_names
)

plt.title("SVM Classification")
plt.show()