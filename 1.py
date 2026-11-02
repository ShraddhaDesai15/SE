# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 1
# Title: Simple Linear Regression using scipy.stats.linregress()
# Dataset: Built-in Diabetes dataset (real-world medical dataset)
# Predict the disease progression measure using one feature: BMI.

# 1. AUTOMATIC INSTALLATION: install only packages that are missing.
import sys
import subprocess
import importlib.util

packages = {"numpy": "numpy", "matplotlib": "matplotlib",
            "scikit-learn": "sklearn", "scipy": "scipy"}

for package, module in packages.items():
    if importlib.util.find_spec(module) is None:
        print("Installing", package)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 2. IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 3. LOAD REAL-WORLD DATASET
data = load_diabetes()
# BMI is one feature. Use one feature because this is SIMPLE linear regression.
bmi_column = list(data.feature_names).index("bmi")
X = data.data[:, bmi_column]
y = data.target

print("Dataset: Diabetes")
print("Number of rows:", len(X))
print("Input feature: BMI")
print("Target: Disease progression measure")

# 4. SPLIT INTO TRAINING AND TESTING DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. USE THE REQUIRED linregress() FUNCTION
result = linregress(X_train, y_train)
slope = result.slope
intercept = result.intercept

# 6. PREDICT TEST VALUES: y = slope*x + intercept
y_pred = slope * X_test + intercept

# 7. DISPLAY RESULTS AND EVALUATION MEASURES
print("\n--- Linear Regression Results ---")
print("Slope:", round(slope, 4))
print("Intercept:", round(intercept, 4))
print("Correlation coefficient (r):", round(result.rvalue, 4))
print("R-squared:", round(r2_score(y_test, y_pred), 4))
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 4))
print("Actual values (first 10):", y_test[:10])
print("Predicted values (first 10):", np.round(y_pred[:10], 2))

# 8. PLOT THE DATA AND REGRESSION LINE
order = np.argsort(X_test)
plt.scatter(X_test, y_test, label="Actual test data")
plt.plot(X_test[order], y_pred[order], color="red", label="Regression line")
plt.xlabel("BMI (scaled dataset feature)")
plt.ylabel("Disease progression measure")
plt.title("Simple Linear Regression using linregress()")
plt.legend()
plt.tight_layout()
plt.show()

# # EXPERIMENT 1: SIMPLE LINEAR REGRESSION

# import sys
# import subprocess

# # Install required libraries automatically
# for package in ["numpy", "matplotlib", "scikit-learn"]:
#     try:
#         __import__({"scikit-learn": "sklearn"}.get(package, package))
#     except ImportError:
#         subprocess.check_call(
#             [sys.executable, "-m", "pip", "install", package]
#         )

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Dataset: Study hours and marks
# X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
# y = np.array([20, 30, 40, 50, 60])

# # Train and predict
# model = LinearRegression()
# model.fit(X, y)
# pred = model.predict(X)

# # Display results
# print("Predicted Marks:", pred)
# print("MSE:", mean_squared_error(y, pred))
# print("R2 Score:", r2_score(y, pred))

# # Plot graph
# plt.scatter(X, y)
# plt.plot(X, pred, color="red")
# plt.xlabel("Study Hours")
# plt.ylabel("Marks")
# plt.title("Linear Regression")
# plt.show()
