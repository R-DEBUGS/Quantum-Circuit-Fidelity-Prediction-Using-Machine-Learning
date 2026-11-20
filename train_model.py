
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# 1. Load dataset
df = pd.read_csv("dataset/quantum_fidelity_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Separate features and target
X = df.drop(columns=["fidelity"])
y = df["fidelity"]

print("\nInput features:")
print(list(X.columns))


# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 4. Define machine learning models
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


# 5. Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# 6. Train and evaluate models
results = []
trained_models = {}

print("\nSTARTING MODEL TRAINING...\n")

for name, model in models.items():

    print("Training:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test values
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        os.path.join("models", filename)
    )

    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("R2 Score:", round(r2, 6))
    print("-" * 40)


# 7. Compare model performance
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))


# 8. Save evaluation results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# 9. Identify best model by test RMSE
best_model_name = results_df.iloc[0]["Model"]

print("\nBEST MODEL:", best_model_name)

# Save best model separately
joblib.dump(
    trained_models[best_model_name],
    "models/best_model.pkl"
)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
print("Trained models saved in models/")
print("Results saved in results/model_comparison.csv")
