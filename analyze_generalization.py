
import pandas as pd
import matplotlib.pyplot as plt

# Load generalization predictions
df = pd.read_csv(
    "results/generalization_predictions.csv"
)

actual = df["actual_fidelity"]
predicted = df["predicted_fidelity"]

# Calculate residuals
residuals = actual - predicted

# Graph 1: Actual vs Predicted
plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.6
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Fidelity")
plt.ylabel("Predicted Fidelity")
plt.title("Higher-Noise Generalization: Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "results/generalization_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# Graph 2: Residuals vs Noise
plt.figure(figsize=(8, 6))

plt.scatter(
    df["single_qubit_error"],
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Single-Qubit Error Probability")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Generalization Residuals vs Noise")

plt.tight_layout()

plt.savefig(
    "results/generalization_residuals.png",
    dpi=300
)

plt.show()


# Print prediction statistics
print("\nGENERALIZATION ERROR ANALYSIS")

print("Mean residual:", residuals.mean())
print("Maximum absolute error:", residuals.abs().max())
print("Minimum actual fidelity:", actual.min())
print("Maximum actual fidelity:", actual.max())

print("\nANALYSIS COMPLETED!")
