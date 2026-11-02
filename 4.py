# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 4
# Title: Multivariate Linear Regression using vectorization and gradient descent
# Dataset: Small example dataset created in this code (house prices).
# Features: area, bedrooms, age. Target: house price in thousands.
# This demonstrates the requested vectorization, cost function, gradient descent,
# model training, SSE, MSE and R-squared.

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 3. CREATE A SMALL EXAMPLE DATASET
# Each row: [area in square feet, number of bedrooms, age of house in years]
X = np.array([
    [800, 2, 15], [900, 2, 10], [1000, 2, 8], [1100, 3, 12],
    [1200, 3, 10], [1300, 3, 7], [1400, 3, 5], [1500, 3, 8],
    [1600, 3, 4], [1700, 4, 6], [1800, 4, 5], [1900, 4, 3],
    [2000, 4, 4], [2100, 4, 2], [2200, 4, 3], [2300, 4, 1],
    [2400, 5, 2], [2500, 5, 1], [2600, 5, 4], [2700, 5, 2],
    [850, 2, 20], [1050, 3, 18], [1250, 3, 14], [1450, 3, 12],
    [1650, 4, 11], [1850, 4, 9], [2050, 4, 7], [2250, 5, 6],
    [2450, 5, 5], [2650, 5, 3]
], dtype=float)

# Price is created from a simple relationship plus small repeatable variation.
rng = np.random.default_rng(42)
y = 50 + 0.12 * X[:, 0] + 18 * X[:, 1] - 1.5 * X[:, 2]
y = y + rng.normal(0, 8, size=len(y))  # small noise; price in thousands

print("Dataset rows:", len(X))
print("Features: Area, Bedrooms, Age")
print("First five feature rows:\n", X[:5])
print("First five prices:", np.round(y[:5], 2))

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. STANDARDIZE INPUT FEATURES AND TARGET TO HELP GRADIENT DESCENT
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

# Add a column of ones for the intercept/bias term.
# This is vectorization: many predictions are calculated with one matrix operation.
X_train_vec = np.c_[np.ones(len(X_train_scaled)), X_train_scaled]
X_test_vec = np.c_[np.ones(len(X_test_scaled)), X_test_scaled]

# 6. COST FUNCTION: J(theta) = sum((predicted - actual)^2) / (2*m)
def cost_function(X_matrix, y_values, theta):
    errors = X_matrix @ theta - y_values
    return np.sum(errors ** 2) / (2 * len(y_values))

# 7. GRADIENT DESCENT: repeatedly update theta to reduce cost.
theta = np.zeros(X_train_vec.shape[1])
learning_rate = 0.05
iterations = 2000
cost_history = []

for i in range(iterations):
    predictions = X_train_vec @ theta
    errors = predictions - y_train_scaled
    gradient = (X_train_vec.T @ errors) / len(y_train_scaled)
    theta = theta - learning_rate * gradient
    cost_history.append(cost_function(X_train_vec, y_train_scaled, theta))

# 8. MODEL TRAINING AND PREDICTION
y_pred_scaled = X_test_vec @ theta
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 9. EVALUATION: SSE, MSE and R-squared
sse = np.sum((y_test - y_pred) ** 2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Multivariate Linear Regression Results ---")
print("Learned parameters (standardized features):", np.round(theta, 4))
print("Final training cost:", round(cost_history[-1], 6))
print("Actual prices:", np.round(y_test, 2))
print("Predicted prices:", np.round(y_pred, 2))
print("SSE (Sum of Squared Errors):", round(sse, 2))
print("MSE (Mean Squared Error):", round(mse, 2))
print("R-squared:", round(r2, 4))

# 10. PLOT COST OVER TRAINING ITERATIONS
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost Reduction")
plt.tight_layout()
plt.show()

# 11. PLOT ACTUAL VS PREDICTED PRICES
plt.scatter(y_test, y_pred)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual House Price (thousands)")
plt.ylabel("Predicted House Price (thousands)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# # EXPERIMENT 4: MULTIVARIATE LINEAR REGRESSION

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
# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # LOAD DATASET
# data = load_diabetes()
# X = data.data
# y = data.target

# # SPLIT DATASET
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # TRAIN MODEL
# model = LinearRegression()
# model.fit(X_train, y_train)

# # PREDICTION
# y_pred = model.predict(X_test)

# # DISPLAY RESULTS
# print("Actual Values:", y_test[:10])
# print("Predicted Values:", y_pred[:10].round(2))
# print("Coefficients:", model.coef_)
# print("Intercept:", round(model.intercept_, 2))
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# # GRAPH
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Multivariate Linear Regression")
# plt.show()