# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ================= CHECK CURRENT FOLDER =================
print("Current Folder:", os.getcwd())
print("Files in Folder:", os.listdir())

# ================= AUTOMATIC CSV LOADER =================
# This will pick the first CSV file in the folder automatically
csv_files = [f for f in os.listdir() if f.endswith(".csv")]

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV file found in this folder! Please add the Boston Housing CSV")
else:
    print("Loading file:", csv_files[0])
    df = pd.read_csv(csv_files[0])

# ================= DATA EXPLORATION =================
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ================= HANDLE MISSING VALUES =================
# Fill missing numeric values with mean (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# ================= FEATURE SELECTION =================
features = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'AGE']
target = 'MEDV'

X = df[features]
y = df[target]

# ================= VISUALIZATION =================
# Scatter plots for each feature vs target
for col in features:
    plt.figure(figsize=(5,4))
    plt.scatter(df[col], y)
    plt.xlabel(col)
    plt.ylabel("Median House Value (MEDV)")
    plt.title(f"{col} vs MEDV")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================= SPLIT TRAIN-TEST =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= BUILD MULTIPLE LINEAR REGRESSION MODEL =================
model = LinearRegression()
model.fit(X_train, y_train)

# ================= PREDICTION =================
y_pred = model.predict(X_test)

# ================= EVALUATION =================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# ================= FEATURE IMPORTANCE =================
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
print("\nFeature Importance (Coefficients):")
print(coefficients)

# ================= ACTUAL vs PREDICTED PLOT =================
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted House Prices")
plt.show()
