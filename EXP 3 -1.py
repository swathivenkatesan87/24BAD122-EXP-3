print("Name : SWATHI")
print("Roll No : 24BAD122")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\StudentsPerformance (1).csv")
# Select Independent and Dependent variables
X = df[["reading score"]].values
y = df["math score"].values
# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
degrees = [2, 3, 4]
train_errors = []
test_errors = []
print("\nPolynomial Regression Results\n")
for d in degrees:
    poly = PolynomialFeatures(degree=d)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    train_errors.append(mean_squared_error(y_train, y_pred_train))
    test_errors.append(mse)

    print("Polynomial Degree :", d)
    print("MSE :", round(mse, 2))
    print("RMSE :", round(np.sqrt(mse), 2))
    print("R2 Score :", round(r2, 3))
    print()

# Ridge Regression (Degree 4)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)
y_ridge_pred = ridge.predict(X_test_poly)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)
ridge_r2 = r2_score(y_test, y_ridge_pred)
print("Ridge Regression Results (Degree 4)")
print("MSE :", round(ridge_mse, 2))
print("RMSE :", round(np.sqrt(ridge_mse), 2))
print("R2 Score :", round(ridge_r2, 3))
plt.figure()
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
X_range_scaled = scaler.transform(X_range)
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train), y_train)

    y_range_pred = model.predict(
        poly.transform(X_range_scaled)
    )

    plt.plot(X_range.flatten(), y_range_pred, label=f"Degree {d}")
plt.scatter(X, y, s=10)
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.title("Polynomial Regression Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='o', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error")
plt.legend()
plt.show()
residuals = y_test - y_ridge_pred

plt.figure()
plt.hist(residuals, bins=20)
plt.xlabel("Residual Value")
plt.title("Residual Distribution (Ridge)")
plt.show()
