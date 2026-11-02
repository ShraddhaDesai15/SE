# EXPERIMENT 4: MULTIVARIATE LINEAR REGRESSION

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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# LOAD DATASET
data = load_diabetes()
X = data.data
y = data.target

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# PREDICTION
y_pred = model.predict(X_test)

# DISPLAY RESULTS
print("Actual Values:", y_test[:10])
print("Predicted Values:", y_pred[:10].round(2))
print("Coefficients:", model.coef_)
print("Intercept:", round(model.intercept_, 2))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# GRAPH
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Multivariate Linear Regression")
plt.show()