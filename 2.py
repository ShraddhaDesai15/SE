# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 2
# Title: Logistic Regression Classification
# Dataset: Built-in Breast Cancer dataset
# Output: Accuracy, confusion matrix, classification report and plot.

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
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# 3. LOAD DATASET
data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset: Breast Cancer")
print("Rows:", X.shape[0], "| Features:", X.shape[1])
print("Classes:", data.target_names)

# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. CREATE PIPELINE: scale features, then train Logistic Regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)
model.fit(X_train, y_train)

# 6. PREDICT TEST CLASSES
y_pred = model.predict(X_test)

# 7. PRINT EVALUATION RESULTS
print("\n--- Logistic Regression Results ---")
print("Actual classes (first 10):", y_test[:10])
print("Predicted classes (first 10):", y_pred[:10])
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))

# 8. DISPLAY CONFUSION MATRIX
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=data.target_names
)
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()

# # EXPERIMENT 2: LOGISTIC REGRESSION

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split

# # Dataset
# X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# # Train model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Display results
# print("Actual Values:", y_test)
# print("Predicted Values:", y_pred)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # Plot confusion matrix
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# plt.title("Logistic Regression Confusion Matrix")
# plt.show()
