
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("dataset/quantum_fidelity_dataset.csv")

# Same noise-based split
train_df = df[df["single_qubit_error"] <= 0.012]
test_df = df[df["single_qubit_error"] > 0.012]

X_train = train_df.drop(columns=["fidelity"])
y_train = train_df["fidelity"]

X_test = test_df.drop(columns=["fidelity"])
y_test = test_df["fidelity"]

# Define models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

results = []

print("TRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

for name, model in models.items():

    print("\nEvaluating:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mean_residual = np.mean(y_test - predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Mean Residual": mean_residual
    })

    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("R2:", round(r2, 6))
    print("Mean Residual:", round(mean_residual, 6))

# Save comparison
results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/generalization_model_comparison.csv",
    index=False
)

print("\nFINAL GENERALIZATION COMPARISON")
print(results_df.to_string(index=False))

print("\nCOMPARISON COMPLETED!")
