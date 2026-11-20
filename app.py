
import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Quantum Fidelity Predictor",
    page_icon="⚛️",
    layout="wide"
)

# ==========================================
# FILE PATHS
# ==========================================

MODEL_PATH = "models/best_model.pkl"
DATASET_PATH = "dataset/quantum_fidelity_dataset.csv"

# ==========================================
# LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

if not os.path.exists(DATASET_PATH):
    st.error("Dataset not found. Generate the dataset first.")
    st.stop()

model = load_model()
df = load_dataset()

FEATURES = [
    "qubit_count",
    "circuit_depth",
    "total_gates",
    "h_count",
    "ry_count",
    "rz_count",
    "cx_count",
    "single_qubit_error",
    "two_qubit_error"
]

# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("⚛️ Quantum Circuit Fidelity Predictor")

st.write(
    "An ML-based system for predicting quantum circuit "
    "fidelity using circuit characteristics and noise parameters."
)

st.info(
    "This application uses a trained XGBoost model "
    "developed using Qiskit noisy circuit simulations."
)

st.divider()

# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test R² Score", "99.02%")

with col2:
    st.metric("Test MAE", "0.011499")

with col3:
    st.metric("Test RMSE", "0.015152")

st.caption(
    "Metrics are from the random 80:20 train-test evaluation "
    "on the regenerated synthetic dataset."
)

st.divider()

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("Enter Quantum Circuit Parameters")

st.caption(
    "Input ranges are based on the generated training dataset."
)

col1, col2 = st.columns(2)

# Dataset-based input limits
def feature_limits(feature):
    return (
        float(df[feature].min()),
        float(df[feature].max())
    )


with col1:

    q_min, q_max = feature_limits("qubit_count")
    d_min, d_max = feature_limits("circuit_depth")

    qubit_count = st.slider(
        "Number of Qubits",
        min_value=int(q_min),
        max_value=int(q_max),
        value=3
    )

    circuit_depth = st.slider(
        "Circuit Depth",
        min_value=int(d_min),
        max_value=int(d_max),
        value=int(df["circuit_depth"].median())
    )

    h_min, h_max = feature_limits("h_count")
    ry_min, ry_max = feature_limits("ry_count")

    h_count = st.number_input(
        "Hadamard Gates (H)",
        min_value=int(h_min),
        max_value=int(h_max),
        value=int(df["h_count"].median())
    )

    ry_count = st.number_input(
        "RY Gates",
        min_value=int(ry_min),
        max_value=int(ry_max),
        value=int(df["ry_count"].median())
    )


with col2:

    rz_min, rz_max = feature_limits("rz_count")
    cx_min, cx_max = feature_limits("cx_count")

    rz_count = st.number_input(
        "RZ Gates",
        min_value=int(rz_min),
        max_value=int(rz_max),
        value=int(df["rz_count"].median())
    )

    cx_count = st.number_input(
        "CNOT Gates",
        min_value=int(cx_min),
        max_value=int(cx_max),
        value=int(df["cx_count"].median())
    )

    se_min, se_max = feature_limits("single_qubit_error")

    single_qubit_error = st.slider(
        "Single-Qubit Error Probability",
        min_value=se_min,
        max_value=se_max,
        value=float(df["single_qubit_error"].median()),
        step=0.001,
        format="%.3f"
    )

    te_min, te_max = feature_limits("two_qubit_error")

    two_qubit_error = st.slider(
        "Two-Qubit Error Probability",
        min_value=te_min,
        max_value=te_max,
        value=float(df["two_qubit_error"].median()),
        step=0.001,
        format="%.3f"
    )

# ==========================================
# CALCULATE TOTAL GATES
# ==========================================

total_gates = (
    h_count + ry_count + rz_count + cx_count
)

st.divider()

st.subheader("Circuit Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Quantum Gates", total_gates)

with col2:
    st.metric("Qubits", qubit_count)

with col3:
    st.metric("Circuit Depth", circuit_depth)

# ==========================================
# CHECK INPUT DISTRIBUTION
# ==========================================

input_values = {
    "qubit_count": qubit_count,
    "circuit_depth": circuit_depth,
    "total_gates": total_gates,
    "h_count": h_count,
    "ry_count": ry_count,
    "rz_count": rz_count,
    "cx_count": cx_count,
    "single_qubit_error": single_qubit_error,
    "two_qubit_error": two_qubit_error
}

out_of_range = []

for feature, value in input_values.items():

    minimum = df[feature].min()
    maximum = df[feature].max()

    if value < minimum or value > maximum:
        out_of_range.append(feature)

if out_of_range:
    st.warning(
        "Some input values are outside the observed training "
        "dataset range: " + ", ".join(out_of_range)
    )

# ==========================================
# PREDICTION
# ==========================================


if st.button("Predict Circuit Fidelity", type="primary"):

    single_gate_total = h_count + ry_count + rz_count

    if total_gates == 0:
        st.error("Please enter at least one quantum gate.")

    elif single_gate_total % qubit_count != 0:
        st.error(
            "Invalid circuit configuration! "
            "The total number of single-qubit gates "
            "(H + RY + RZ) must be divisible by the number of qubits."
        )

    else:
        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction Results")

        if not 0 <= prediction <= 1:
            st.error(
                "Prediction is outside the physical fidelity range. "
                "This input may be outside the model's reliable range."
            )

        else:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Fidelity",
                    f"{prediction:.4f}"
                )

            with col2:
                st.metric(
                    "Fidelity Percentage",
                    f"{prediction * 100:.2f}%"
                )

            st.progress(prediction)

            if prediction >= 0.90:
                st.success("The model predicts high circuit fidelity.")
            elif prediction >= 0.75:
                st.info("The model predicts moderate circuit fidelity.")
            else:
                st.warning("The model predicts relatively low circuit fidelity.")

            with st.expander("View prediction input parameters"):
                st.dataframe(input_data, use_container_width=True)

        st.caption(
            "This is an ML prediction based on synthetic Qiskit "
            "simulation data, not a measurement from real quantum hardware."
        )


    if total_gates == 0:

        st.error("Please enter at least one quantum gate.")

    else:

        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction Results")

        if not 0 <= prediction <= 1:

            st.error(
                "The model produced a value outside the physical "
                "fidelity range. Treat this prediction as unreliable."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Fidelity",
                    f"{prediction:.4f}"
                )

            with col2:
                st.metric(
                    "Fidelity Percentage",
                    f"{prediction * 100:.2f}%"
                )

            st.progress(prediction)

            if prediction >= 0.90:

                st.success(
                    "The model predicts high circuit fidelity."
                )

            elif prediction >= 0.75:

                st.info(
                    "The model predicts moderate circuit fidelity."
                )

            else:

                st.warning(
                    "The model predicts relatively low circuit fidelity."
                )

            # Input summary
            with st.expander("View prediction input parameters"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        st.caption(
            "This is an ML prediction based on a synthetic "
            "Qiskit training dataset, not a measurement from "
            "a real quantum computer."
        )

# ==========================================
# ABOUT THE PROJECT
# ==========================================

st.divider()

st.subheader("About This Project")

st.write("""
This project uses Qiskit and Qiskit Aer to generate quantum
circuits and simulate depolarizing noise.

Machine learning models are trained to predict quantum
circuit fidelity using circuit characteristics and
noise parameters.

Three regression models are compared:

- Linear Regression
- Random Forest
- XGBoost

XGBoost achieves the best performance on the random
test-set evaluation and is selected for the prediction application.
""")

with st.expander("Project limitations"):

    st.write("""
    - The training dataset is synthetically generated using Qiskit simulations.
    - Predictions are not measurements from real quantum hardware.
    - Manually entered features may not represent a physically valid circuit.
    - Performance on unseen noise conditions can differ from random test-set performance.
    - Model predictions should be interpreted within the training data distribution.
    """)
