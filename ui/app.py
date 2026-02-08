import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention


# Utility: Generate synthetic data
def generate_sequence(
    seq_len = 75,
    init_speed = 60,
    aggressiveness = 0.5,
    noise_level = 0.05
):
    speed = init_speed
    accel = 0.0
    brake = 0.0

    data = []

    target_brake = np.clip(aggressiveness, 0.1, 1.0)

    for _ in range(seq_len):
        brake += (target_brake - brake) * np.random.uniform(0.03, 0.08)
        brake += np.random.normal(0, noise_level)
        brake = np.clip(brake, 0, 1)

        accel = -brake * np.random.uniform(2.0, 3.5)
        accel += np.random.normal(0, noise_level)

        speed += accel * 0.1
        speed = max(speed, 0)

        data.append([speed, accel, brake])

    return np.array(data)


# Load trained model
@st.cache_resource
def load_model():
    model = MultitaskLSTMCNNAttention()
    model.load_state_dict(
        torch.load("models/final_multitask_model.pth", map_location="cpu")
    )
    model.eval()

    return model


# Streamlit UI
st.set_page_config(page_title = "Braking Intention Predictor", layout = "wide")

st.title("🚗 Braking Intention Prediction (Multitask Learning)")
st.write(
    "Predicts **driver braking intention** and **brake intensity** "
    "from vehicle time-series data using a deep learning model."
)

# Sidebar inputs
st.sidebar.header("Input Controls")

init_speed = st.sidebar.slider(
    "Initial Speed (km/h)", 20, 120, 60
)

aggressiveness = st.sidebar.slider(
    "Braking Aggressiveness", 0.1, 1.0, 0.5
)

noise_level = st.sidebar.slider(
    "Noise Level", 0.0, 0.2, 0.05
)

run_button = st.sidebar.button("Run Prediction")

# Main logic
if run_button:
    # Generate data
    sequence = generate_sequence(
        init_speed = init_speed,
        aggressiveness = aggressiveness,
        noise_level = noise_level
    )

    # Plot time-series
    st.subheader("Input Time-Series")
    fig, axs = plt.subplots(3, 1, figsize = (10, 6), sharex = True)

    axs[0].plot(sequence[:, 0])
    axs[0].set_ylabel("Speed")

    axs[1].plot(sequence[:, 1])
    axs[1].set_ylabel("Acceleration")

    axs[2].plot(sequence[:, 2])
    axs[2].set_ylabel("Brake Pedal")
    axs[2].set_xlabel("Time Step")

    st.pyplot(fig)

    # Model inference
    model = load_model()

    x = torch.tensor(sequence, dtype = torch.float32).unsqueeze(0)

    with torch.no_grad():
        class_logits, intensity = model(x)
        probs = torch.softmax(class_logits, dim = 1).numpy()[0]
        pred_class = probs.argmax()
        pred_intensity = intensity.item()

    class_names = ["Light Braking", "Normal Braking", "Emergency Braking"]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Braking Intention",
            value=class_names[pred_class]
        )

    with col2:
        st.metric(
            label = "Predicted Brake Intensity",
            value = f"{pred_intensity:.2f}"
        )

    st.write("### Class Probabilities")
    for name, p in zip(class_names, probs):
        st.write(f"- **{name}**: {p:.2f}")
