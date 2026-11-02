# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 3
# Title: Decision Tree (Gini), Random Forest, Bagging and Boosting
# Dataset: Built-in Breast Cancer classification dataset.
# Compares ensemble models with a Decision Tree baseline.

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 3. LOAD AND SPLIT DATASET
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Dataset: Breast Cancer")
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# 4. CREATE MODELS
# Decision Tree uses Gini index and acts as the baseline model.
tree = DecisionTreeClassifier(criterion="gini", random_state=42)

# Random Forest combines multiple decision trees.
forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Bagging explicitly trains many copies of a base decision tree on samples.
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(criterion="gini", random_state=42),
    n_estimators=100, random_state=42
)

# AdaBoost uses shallow trees as weak learners and combines them sequentially.
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50, learning_rate=1.0, random_state=42
)

models = {
    "Decision Tree (Gini)": tree,
    "Random Forest": forest,
    "Bagging": bagging,
    "AdaBoost": boosting
}

# 5. TRAIN, PREDICT AND EVALUATE EACH MODEL
accuracies = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    accuracies[name] = accuracy_score(y_test, pred)

    print("\n---", name, "---")
    print("Accuracy:", round(accuracies[name], 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    # Plot a confusion matrix for each model.
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=data.target_names
    )
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# 6. COMPARE ALL ACCURACIES
print("\n--- Accuracy Comparison ---")
for name, score in accuracies.items():
    print(f"{name}: {score:.4f} ({score * 100:.2f}%)")

plt.figure(figsize=(9, 5))
plt.bar(list(accuracies.keys()), list(accuracies.values()))
plt.ylabel("Accuracy")
plt.title("Decision Tree vs Random Forest vs Bagging vs AdaBoost")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# Comparison notes:
# Decision Tree is the non-ensemble baseline.
# Compare Bagging and Random Forest against the Decision Tree.
# Compare AdaBoost against the Decision Tree.

# # EXPERIMENT 3: BOOSTING AND BAGGING

# import sys
# import subprocess
# import importlib.util

# # AUTOMATICALLY INSTALL REQUIRED LIBRARIES
# packages = {
#     "numpy": "numpy",
#     "matplotlib": "matplotlib",
#     "scikit-learn": "sklearn"
# }

# for package, module in packages.items():
#     if importlib.util.find_spec(module) is None:
#         print("Installing", package, "...")
#         subprocess.check_call(
#             [sys.executable, "-m", "pip", "install", package]
#         )

# # IMPORT LIBRARIES
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix

# # LOAD DATASET
# data = load_breast_cancer()
# X = data.data
# y = data.target

# # SPLIT DATASET
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # BAGGING: RANDOM FOREST
# bagging_model = RandomForestClassifier(
#     n_estimators=100, random_state=42
# )
# bagging_model.fit(X_train, y_train)

# bagging_pred = bagging_model.predict(X_test)

# # BOOSTING: ADABOOST
# boosting_model = AdaBoostClassifier(
#     n_estimators=50, random_state=42
# )
# boosting_model.fit(X_train, y_train)

# boosting_pred = boosting_model.predict(X_test)

# # DISPLAY RESULTS
# bagging_accuracy = accuracy_score(y_test, bagging_pred)
# boosting_accuracy = accuracy_score(y_test, boosting_pred)

# print("\n===== BAGGING: RANDOM FOREST =====")
# print("Accuracy:", round(bagging_accuracy, 4))
# print("Confusion Matrix:\n", confusion_matrix(y_test, bagging_pred))

# print("\n===== BOOSTING: ADABOOST =====")
# print("Accuracy:", round(boosting_accuracy, 4))
# print("Confusion Matrix:\n", confusion_matrix(y_test, boosting_pred))

# # COMPARE MODELS GRAPHICALLY
# plt.bar(
#     ["Random Forest (Bagging)", "AdaBoost (Boosting)"],
#     [bagging_accuracy, boosting_accuracy]
# )

# plt.ylabel("Accuracy")
# plt.title("Comparison of Bagging and Boosting")
# plt.ylim(0, 1.05)
# plt.show()