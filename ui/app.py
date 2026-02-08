# Ensure project root is in sys.path for module imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime



# Different Scenario Presets
SCENARIO_PRESETS = {
    "Manual (Custom)": None,
    "City Light Brake": (40, 0.25, 0.05),
    "Highway Gentle Slowdown": (90, 0.35, 0.05),
    "Sudden Emergency Brake": (80, 0.85, 0.10),
    "Stop-and-Go Traffic": (35, 0.55, 0.08),
    "Aggressive Driver": (70, 0.75, 0.12),
}


# Synthetic Data Generator
def generate_sequence(seq_len=75, init_speed=60, aggressiveness=0.5, noise_level=0.05):
    speed = init_speed
    brake = 0.0
    data = []

    for _ in range(seq_len):
        brake += (aggressiveness - brake) * np.random.uniform(0.03, 0.08)
        brake += np.random.normal(0, noise_level)
        brake = np.clip(brake, 0, 1)

        accel = -brake * np.random.uniform(2.0, 3.5)
        accel += np.random.normal(0, noise_level)

        speed += accel * 0.1
        speed = max(speed, 0)

        data.append([speed, accel, brake])

    return np.array(data)


# Load Model
@st.cache_resource
def load_model():
    from models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
    model = MultitaskLSTMCNNAttention()
    model.load_state_dict(
        torch.load("models/final_multitask_model.pth", map_location="cpu")
    )
    model.eval()
    return model


# Page Config
st.set_page_config(page_title="Braking Intention Prediction", layout="wide")


# Title (Centered)
st.markdown(
    """
    <h1 style="text-align:center;">Braking Intention Prediction</h1>
    <p style="text-align:center; color:#b0b0b0;">
    Predicts driver braking intention and brake intensity from vehicle time-series data.
    </p>
    """,
    unsafe_allow_html = True
)


# Top Information Dropdowns
c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("What does this app do?"):
        st.write(
            "This application predicts **driver braking intention** "
            "(Light, Normal, Emergency) and **brake intensity** "
            "using a deep learning model trained on vehicle time-series data."
        )

with c2:
    with st.expander("What data is used?"):
        st.write(
            "- Vehicle speed\n"
            "- Acceleration (deceleration)\n"
            "- Brake pedal input\n\n"
            "These signals simulate realistic braking behavior over time."
        )

with c3:
    with st.expander("How to interpret results?"):
        st.write(
            "- 🟢 **Light Braking**: Low risk\n"
            "- 🟡 **Normal Braking**: Moderate braking\n"
            "- 🔴 **Emergency Braking**: Sudden / high-risk braking"
        )

st.divider()


# Scenario Preset Selector
st.subheader("Scenario Preset")

selected_scenario = st.selectbox(
    "Choose a predefined driving scenario (or use Manual for sliders):",
    options=list(SCENARIO_PRESETS.keys())
)

if selected_scenario != "Manual (Custom)":
    default_speed, default_aggr, default_noise = SCENARIO_PRESETS[selected_scenario]
else:
    default_speed, default_aggr, default_noise = 48, 0.5, 0.05


# Input Controls
st.subheader("Input Controls")

ic1, ic2, ic3, ic4 = st.columns([1, 1, 1, 0.8])

with ic1:
    init_speed = st.slider(
        "Initial Speed (km/h)", 20, 120, int(default_speed)
    )

with ic2:
    aggressiveness = st.slider(
        "Braking Aggressiveness", 0.1, 1.0, float(default_aggr)
    )

with ic3:
    noise_level = st.slider(
        "Noise Level", 0.0, 0.2, float(default_noise)
    )

with ic4:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Prediction", use_container_width=True)

st.divider()


# Main Layout
left, right = st.columns([1.5, 1])

if run:
    sequence = generate_sequence(
        init_speed=init_speed,
        aggressiveness=aggressiveness,
        noise_level=noise_level
    )

    # LEFT: Time-Series Plots
    with left:
        st.subheader("Input Time-Series Signals")

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        axs[0].plot(sequence[:, 0], color="#1f77b4")
        axs[0].set_ylabel("Speed")

        axs[1].plot(sequence[:, 1], color="#ff7f0e")
        axs[1].set_ylabel("Acceleration")

        axs[2].plot(sequence[:, 2], color="#d62728")
        axs[2].set_ylabel("Brake Pedal")
        axs[2].set_xlabel("Time Step")

        for ax in axs:
            ax.grid(alpha=0.3)

        st.pyplot(fig)

        with st.expander("What do these signals indicate?"):
            st.write(
                "- **Speed** decreases gradually, indicating sustained braking.\n"
                "- **Acceleration** remains negative, confirming deceleration.\n"
                "- **Brake pedal** input increases and stabilizes, reflecting driver intent."
            )

    # RIGHT: Predictions
    with right:
        model = load_model()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, intensity = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        classes = ["Light Braking", "Normal Braking", "Emergency Braking"]
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        emojis = ["🟢", "🟡", "🔴"]

        pred = probs.argmax()

        st.subheader("Prediction Results")

        st.markdown(
            f"""
            <div style="
                background:{colors[pred]};
                padding:20px;
                border-radius:12px;
                text-align:center;
                font-size:22px;
                font-weight:bold;">
                {emojis[pred]} {classes[pred]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Predicted Brake Intensity", f"{intensity.item():.2f}")

        # Pie Chart
        fig2, ax2 = plt.subplots()

        wedges, _, _ = ax2.pie(
            probs,
            autopct="%1.0f%%",
            startangle=90,
            colors=colors
        )

        ax2.legend(
            wedges,
            classes,
            title="Braking Classes",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )

        ax2.axis("equal")
        st.pyplot(fig2)

        # Export Prediction
        report = {
            "timestamp": datetime.now().isoformat(),
            "scenario": selected_scenario,
            "inputs": {
                "initial_speed_kmh": init_speed,
                "braking_aggressiveness": aggressiveness,
                "noise_level": noise_level,
            },
            "prediction": {
                "braking_intention": classes[pred],
                "brake_intensity": float(intensity.item()),
                "class_probabilities": {
                    classes[i]: float(probs[i]) for i in range(3)
                }
            }
        }

        st.download_button(
            label="📥 Download Prediction Report",
            data=json.dumps(report, indent=4),
            file_name="braking_prediction_report.json",
            mime="application/json"
        )

