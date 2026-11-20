
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/quantum_fidelity_dataset.csv")

print("DATASET INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# 1. Fidelity distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["fidelity"], bins=15, kde=True)
plt.title("Distribution of Quantum Circuit Fidelity")
plt.xlabel("Fidelity")
plt.ylabel("Number of Circuits")
plt.tight_layout()
plt.savefig("dataset/fidelity_distribution.png")
plt.show()

# 2. Circuit depth vs fidelity
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="circuit_depth",
    y="fidelity"
)
plt.title("Circuit Depth vs Fidelity")
plt.tight_layout()
plt.savefig("dataset/depth_vs_fidelity.png")
plt.show()

# 3. Noise vs fidelity
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="single_qubit_error",
    y="fidelity",
    label="Single-qubit error"
)
plt.title("Single-Qubit Error vs Fidelity")
plt.tight_layout()
plt.savefig("dataset/noise_vs_fidelity.png")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("dataset/correlation_heatmap.png")
plt.show()

print("\nDATASET ANALYSIS COMPLETED!")