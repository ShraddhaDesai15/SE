
# EXPERIMENT 1: SIMPLE LINEAR REGRESSION

import sys
import subprocess

# Install required libraries automatically
for package in ["numpy", "matplotlib", "scikit-learn"]:
    try:
        __import__({"scikit-learn": "sklearn"}.get(package, package))
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset: Study hours and marks
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([20, 30, 40, 50, 60])

# Train and predict
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

# Display results
print("Predicted Marks:", pred)
print("MSE:", mean_squared_error(y, pred))
print("R2 Score:", r2_score(y, pred))

# Plot graph
plt.scatter(X, y)
plt.plot(X, pred, color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.show()
