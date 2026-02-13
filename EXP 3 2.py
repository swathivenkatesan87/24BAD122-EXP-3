print("Name : SWATHI")
print("Roll No : 24BAD122")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
# 2. Load Auto MPG Dataset (LOCAL PATH)
file_path =( r"C:\Users\DELL\Downloads\auto-mpg.csv " ) 
df = pd.read_csv(file_path)
print("First 5 rows of dataset:")
print(df.head())
# 3. Select horsepower as independent variable
df = df[['horsepower', 'mpg']]
# 4. Handle Missing Values
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(inplace=True)
X = df[['horsepower']].values
y = df['mpg'].values
# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
degrees = [2, 3, 4]
results = {}
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    scaler = StandardScaler()
    X_poly_train = scaler.fit_transform(X_poly_train)
    X_poly_test = scaler.transform(X_poly_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    results[d] = {
        "model": model,
        "poly": poly,
        "scaler": scaler,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "train_error": mean_squared_error(y_train, y_train_pred),
        "test_error": mse
    }
    # Print Results
print("\nModel Performance Comparison:\n")
for d in degrees:
    print(f"Degree {d}:")
    print(f"MSE  : {results[d]['mse']:.4f}")
    print(f"RMSE : {results[d]['rmse']:.4f}")
    print(f"R2   : {results[d]['r2']:.4f}")
    print("-" * 30)
# Ridge Regression (Degree 4)
poly = PolynomialFeatures(degree=4)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
scaler = StandardScaler()
X_poly_train = scaler.fit_transform(X_poly_train)
X_poly_test = scaler.transform(X_poly_test)
ridge = Ridge(alpha=10)
ridge.fit(X_poly_train, y_train)
ridge_pred = ridge.predict(X_poly_test)
print("\nRidge Regression (Degree 4):")
print("MSE  :", mean_squared_error(y_test, ridge_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("R2   :", r2_score(y_test, ridge_pred))
# VISUALIZATION (3 GRAPHS)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.figure(figsize=(15, 5))
# Graph 1
plt.subplot(1, 3, 1)
plt.scatter(X, y)
for d in degrees:
    poly = results[d]["poly"]
    scaler = results[d]["scaler"]
    model = results[d]["model"]

    X_range_poly = poly.transform(X_range)
    X_range_poly = scaler.transform(X_range_poly)
    y_range_pred = model.predict(X_range_poly)

    plt.plot(X_range, y_range_pred)
plt.title("Polynomial Curve Fitting")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
# Graph 2
plt.subplot(1, 3, 2)
train_errors = [results[d]["train_error"] for d in degrees]
test_errors = [results[d]["test_error"] for d in degrees]
plt.plot(degrees, train_errors, marker='o')
plt.plot(degrees, test_errors, marker='o')
plt.title("Training vs Testing Error")
plt.xlabel("Degree")
plt.ylabel("MSE")
# Graph 3
plt.subplot(1, 3, 3)
r2_scores = [results[d]["r2"] for d in degrees]
plt.plot(degrees, r2_scores, marker='o')
plt.title("Model Complexity vs R2 Score")
plt.xlabel("Degree")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.show()
