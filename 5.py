# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 5
# Title: SVM classification and hyperparameter tuning using GridSearchCV
# Dataset: Built-in Breast Cancer dataset.

# 1. AUTOMATIC INSTALLATION
import sys
import subprocess
import importlib.util

packages = {"numpy": "numpy", "pandas": "pandas",
            "matplotlib": "matplotlib", "scikit-learn": "sklearn"}

for package, module in packages.items():
    if importlib.util.find_spec(module) is None:
        print("Installing", package)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 2. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# 3. LOAD DATASET AND SHOW INFORMATION
data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

print("--- Dataset Information ---")
print("Rows and columns:", df.shape)
print("Class names:", data.target_names)
print("\nFirst five rows:\n", df.head())
print("\nMissing values in each column:\n", df.isnull().sum())
print("Total missing values:", int(df.isnull().sum().sum()))

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. TRAIN BASIC SVM
basic_model = make_pipeline(StandardScaler(), SVC())
basic_model.fit(X_train, y_train)
basic_pred = basic_model.predict(X_test)
basic_accuracy = accuracy_score(y_test, basic_pred)

print("\n--- Basic SVM ---")
print("Accuracy:", round(basic_accuracy, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, basic_pred))
print("Classification Report:\n",
      classification_report(y_test, basic_pred, target_names=data.target_names))

ConfusionMatrixDisplay.from_predictions(
    y_test, basic_pred, display_labels=data.target_names
)
plt.title("Basic SVM - Confusion Matrix")
plt.tight_layout()
plt.show()

# 6. HYPERPARAMETER TUNING USING GRIDSEARCHCV
# StandardScaler is fitted inside each cross-validation fold through the pipeline.
pipeline = make_pipeline(StandardScaler(), SVC())
parameters = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"]
}
grid = GridSearchCV(pipeline, parameters, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

# 7. EVALUATE THE BEST MODEL
best_model = grid.best_estimator_
grid_pred = best_model.predict(X_test)
grid_accuracy = accuracy_score(y_test, grid_pred)

print("\n--- SVM after GridSearchCV ---")
print("Best parameters:", grid.best_params_)
print("Best cross-validation accuracy:", round(grid.best_score_, 4))
print("Test accuracy:", round(grid_accuracy, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, grid_pred))
print("Classification Report:\n",
      classification_report(y_test, grid_pred, target_names=data.target_names))

ConfusionMatrixDisplay.from_predictions(
    y_test, grid_pred, display_labels=data.target_names
)
plt.title("Tuned SVM - Confusion Matrix")
plt.tight_layout()
plt.show()

# 8. COMPARE BASIC AND TUNED SVM
print("\n--- Final Comparison ---")
print(f"Basic SVM accuracy: {basic_accuracy:.4f} ({basic_accuracy * 100:.2f}%)")
print(f"Tuned SVM accuracy: {grid_accuracy:.4f} ({grid_accuracy * 100:.2f}%)")

plt.bar(["Basic SVM", "GridSearchCV SVM"], [basic_accuracy, grid_accuracy])
plt.ylabel("Test accuracy")
plt.title("Basic SVM vs GridSearchCV SVM")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# # EXPERIMENT 5: SUPPORT VECTOR MACHINE

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
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay

# # LOAD DATASET
# data = load_breast_cancer()
# X = data.data
# y = data.target

# # SPLIT DATASET
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # BASIC SVM MODEL
# model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("BASIC SVM ACCURACY:", accuracy_score(y_test, y_pred))
# print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))

# # PARAMETER TUNING
# parameters = {
#     "svc__C": [0.1, 1, 10],
#     "svc__kernel": ["linear", "rbf"]
# }

# grid = GridSearchCV(
#     make_pipeline(StandardScaler(), SVC()),
#     parameters,
#     cv=5,
#     scoring="accuracy"
# )

# grid.fit(X_train, y_train)

# best_model = grid.best_estimator_
# tuned_pred = best_model.predict(X_test)

# print("\nBEST PARAMETERS:", grid.best_params_)
# print("TUNED SVM ACCURACY:", accuracy_score(y_test, tuned_pred))
# print("TUNED CONFUSION MATRIX:\n", confusion_matrix(y_test, tuned_pred))

# # CONFUSION MATRIX GRAPH
# ConfusionMatrixDisplay.from_predictions(
#     y_test, tuned_pred, display_labels=data.target_names
# )

# plt.title("SVM Classification")
# plt.show()