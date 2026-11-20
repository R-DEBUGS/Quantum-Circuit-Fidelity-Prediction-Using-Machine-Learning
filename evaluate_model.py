
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv("dataset/quantum_fidelity_dataset.csv")

X = df.drop(columns=["fidelity"])
y = df["fidelity"]

# Use exactly the same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

os.makedirs("results", exist_ok=True)

# 2. Load trained models
model_files = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

predictions = {}

for name, path in model_files.items():
    model = joblib.load(path)
    predictions[name] = model.predict(X_test)

# 3. Actual vs predicted for all models
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, pred) in zip(axes, predictions.items()):

    ax.scatter(y_test, pred, alpha=0.7)

    # Perfect prediction reference line
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    ax.set_title(name)
    ax.set_xlabel("Actual Fidelity")
    ax.set_ylabel("Predicted Fidelity")

plt.suptitle("Actual vs Predicted Quantum Circuit Fidelity")
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=300)
plt.show()

# 4. Residual plot for XGBoost
xgb_pred = predictions["XGBoost"]
residuals = y_test - xgb_pred

plt.figure(figsize=(8, 5))
plt.scatter(xgb_pred, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.title("XGBoost Residual Analysis")
plt.xlabel("Predicted Fidelity")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig("results/xgboost_residuals.png", dpi=300)
plt.show()

# 5. XGBoost feature importance
xgb_model = joblib.load("models/xgboost.pkl")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(9, 6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.show()

print("\nXGBOOST FEATURE IMPORTANCE")
print(importance.to_string(index=False))

# 6. Model comparison
comparison = pd.read_csv("results/model_comparison.csv")

plt.figure(figsize=(9, 5))
sns.barplot(
    data=comparison,
    x="Model",
    y="R2 Score"
)
plt.title("Machine Learning Model Comparison (R2 Score)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=300)
plt.show()

print("\nMODEL EVALUATION COMPLETED!")
print("All graphs saved inside results/")
