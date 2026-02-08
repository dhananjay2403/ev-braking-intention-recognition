import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention



# Utility: Generate synthetic data
def generate_sequence(seq_len = 75, init_speed = 60, aggressiveness = 0.5, noise_level = 0.05):
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



# Load model
@st.cache_resource
def load_model():
    model = MultitaskLSTMCNNAttention()
    model.load_state_dict(torch.load("models/final_multitask_model.pth", map_location="cpu"))
    model.eval()
    return model


# Page config
st.set_page_config(page_title = "Braking Intention Prediction", layout = "wide")

st.markdown(
    "<h1 style='text-align: center;'>🚗 Braking Intention Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align: center; color: gray; font-size: 16px;'>"
    "Predicts driver braking intention and brake intensity from vehicle time-series data."
    "</div>",
    unsafe_allow_html = True
)

# Add vertical space between caption and dropdowns
st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html = True)

# Info dropdowns
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    with st.expander("What does this app do?"):
        st.write(
            "This application predicts **driver braking intention** "
            "(Light, Normal, Emergency) and **brake intensity** "
            "using a deep learning model trained on time-series vehicle data."
        )

with col_info2:
    with st.expander("What data is used?"):
        st.write(
            "The model uses synthetic time-series signals:\n"
            "- Vehicle speed\n"
            "- Acceleration\n"
            "- Brake pedal input\n\n"
            "These simulate realistic braking behavior."
        )

with col_info3:
    with st.expander("How to interpret results?"):
        st.write(
            "- **Green**: Light braking (low risk)\n"
            "- **Yellow**: Normal braking\n"
            "- **Red**: Emergency braking (high risk)"
        )

st.divider()


# Input controls (MAIN PAGE)
st.subheader("Input Controls")

col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 1, 0.7])

with col_ctrl1:
    init_speed = st.slider("Initial Speed (km/h)", 20, 120, 60)

with col_ctrl2:
    aggressiveness = st.slider("Braking Aggressiveness", 0.1, 1.0, 0.5)

with col_ctrl3:
    noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05)

with col_ctrl4:
    run_button = st.button("Run Prediction", use_container_width=True)

st.divider()


# Main content layout
left_col, right_col = st.columns([1.4, 1])

if run_button:
    sequence = generate_sequence(init_speed=init_speed,
                                 aggressiveness=aggressiveness,
                                 noise_level=noise_level)

    # LEFT: Graphs
    with left_col:
        st.subheader("Input Time-Series Signals")

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        axs[0].plot(sequence[:, 0], color="tab:blue")
        axs[0].set_ylabel("Speed")
        axs[0].grid(alpha=0.3)

        axs[1].plot(sequence[:, 1], color="tab:orange")
        axs[1].set_ylabel("Acceleration")
        axs[1].grid(alpha=0.3)

        axs[2].plot(sequence[:, 2], color="tab:red")
        axs[2].set_ylabel("Brake Pedal")
        axs[2].set_xlabel("Time Step")
        axs[2].grid(alpha=0.3)

        st.pyplot(fig)

        with st.expander("What do these graphs mean?"):
            st.write(
                "- **Speed** shows how the vehicle slows down over time.\n"
                "- **Acceleration** reflects deceleration due to braking.\n"
                "- **Brake Pedal** shows how strongly the driver presses the brake."
            )

    # RIGHT: Prediction results
    with right_col:
        model = load_model()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            class_logits, intensity = model(x)
            probs = torch.softmax(class_logits, dim=1).numpy()[0]
            pred_class = probs.argmax()
            pred_intensity = intensity.item()

        class_names = ["Light Braking", "Normal Braking", "Emergency Braking"]
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

        st.subheader("Prediction Results")

        st.markdown(
            f"""
            <div style="
                background-color:{colors[pred_class]};
                padding:20px;
                border-radius:10px;
                color:black;
                font-size:22px;
                text-align:center;
                font-weight:bold;">
                {class_names[pred_class]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("Predicted Brake Intensity", f"{pred_intensity:.2f}")

        st.write("### Class Probabilities")
        for name, p in zip(class_names, probs):
            st.write(f"- **{name}**: {p:.2f}")
