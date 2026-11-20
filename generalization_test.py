
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("dataset/quantum_fidelity_dataset.csv")

# Define training and test conditions
train_df = df[
    df["single_qubit_error"] <= 0.012
].copy()

test_df = df[
    df["single_qubit_error"] > 0.012
].copy()

# Separate features and target
X_train = train_df.drop(columns=["fidelity"])
y_train = train_df["fidelity"]

X_test = test_df.drop(columns=["fidelity"])
y_test = test_df["fidelity"]

print("TRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Training or testing dataset is empty.")

# Train XGBoost
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost on lower-noise circuits...")

model.fit(X_train, y_train)

# Predict higher-noise circuits
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nGENERALIZATION TEST RESULTS")

print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("R2 Score:", round(r2, 6))

print("\nTraining noise range:")
print(
    train_df["single_qubit_error"].min(),
    "to",
    train_df["single_qubit_error"].max()
)

print("\nTesting noise range:")
print(
    test_df["single_qubit_error"].min(),
    "to",
    test_df["single_qubit_error"].max()
)

# Save results
result = pd.DataFrame({
    "actual_fidelity": y_test.values,
    "predicted_fidelity": predictions,
    "single_qubit_error": X_test[
        "single_qubit_error"
    ].values
})

result.to_csv(
    "results/generalization_predictions.csv",
    index=False
)

print("\nGeneralization predictions saved!")