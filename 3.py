# EXPERIMENT 3: BOOSTING AND BAGGING

import sys
import subprocess
import importlib.util

# AUTOMATICALLY INSTALL REQUIRED LIBRARIES
packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn"
}

for package, module in packages.items():
    if importlib.util.find_spec(module) is None:
        print("Installing", package, "...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )

# IMPORT LIBRARIES
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# LOAD DATASET
data = load_breast_cancer()
X = data.data
y = data.target

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# BAGGING: RANDOM FOREST
bagging_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
bagging_model.fit(X_train, y_train)

bagging_pred = bagging_model.predict(X_test)

# BOOSTING: ADABOOST
boosting_model = AdaBoostClassifier(
    n_estimators=50, random_state=42
)
boosting_model.fit(X_train, y_train)

boosting_pred = boosting_model.predict(X_test)

# DISPLAY RESULTS
bagging_accuracy = accuracy_score(y_test, bagging_pred)
boosting_accuracy = accuracy_score(y_test, boosting_pred)

print("\n===== BAGGING: RANDOM FOREST =====")
print("Accuracy:", round(bagging_accuracy, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, bagging_pred))

print("\n===== BOOSTING: ADABOOST =====")
print("Accuracy:", round(boosting_accuracy, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, boosting_pred))

# COMPARE MODELS GRAPHICALLY
plt.bar(
    ["Random Forest (Bagging)", "AdaBoost (Boosting)"],
    [bagging_accuracy, boosting_accuracy]
)

plt.ylabel("Accuracy")
plt.title("Comparison of Bagging and Boosting")
plt.ylim(0, 1.05)
plt.show()