# app.py

# --- Imports ---
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import random
import joblib
from torch_geometric.data import Data
import plotly.graph_objects as go

# These imports should work correctly if the files are in the src directory
from src.model import GraphAutoencoder
from src.build_graph import build_graph

# --- Paths ---
MODEL_PATH = "outputs/models/graph_autoencoder.pt"
SCALER_PATH = "outputs/models/scaler.joblib"
TRAIN_DATA_PATH = "data/processed/train.csv"

# --- Load Pre-trained Model & Dependencies ---
@st.cache_resource
def load_resources():
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        feature_names = train_df.columns.tolist()
        num_features = len(feature_names)
        
        edge_index = build_graph(train_df.values)
        
        model = GraphAutoencoder(num_features)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        
        scaler = joblib.load(SCALER_PATH)

        x_train = torch.tensor(train_df.values, dtype=torch.float)
        with torch.no_grad():
            reconstructed_train, _ = model(x_train, edge_index)
        train_scores = torch.mean((x_train - reconstructed_train) ** 2, dim=1).numpy()
        threshold = np.percentile(train_scores, 95)
        
        return model, scaler, feature_names, edge_index, threshold
    except FileNotFoundError as e:
        st.error(f"Error: Missing file. Please ensure all required files exist: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        st.stop()

model, scaler, feature_names, edge_index, threshold = load_resources()

# --- Normal Ranges ---
normal_ranges = {
    "ReactorFeedRateStream6": (100, 300),
    "ReactorPressurekPagauge": (1000, 2000),
    "ReactorLevel": (40, 60),
    "ReactorTemperatureDegC": (200, 300),
    "PurgeRateStream9": (5, 20),
    "ProductSepTempDegC": (150, 250),
    "StripperTemperatureDegC": (100, 200),
    "CompressorWorkkW": (500, 800),
}

# --- Recommendation Rules ---
recommendation_rules = {
    tuple(sorted(("ReactorPressurekPagauge", "ReactorTemperatureDegC"))): [
        "ACTION: Verify cooling system is active and check for outlet blockages.",
        "ACTION: Prepare for potential controlled shutdown if pressure continues to rise."
    ],
    tuple(sorted(("CompressorWorkkW", "ReactorPressurekPagauge"))): [
        "ACTION: Inspect compressor for signs of surge or mechanical fouling.",
        "ACTION: Check recycle valve position and functionality.",
        "ACTION: Monitor compressor bearing temperatures for overheating."
    ],
    "default": ["ACTION: General anomaly detected. Operator to review all process variables."]
}

# --- Sidebar ---
st.sidebar.title("📘 About This Project")
st.sidebar.info(
    """
This system monitors industrial processes by detecting anomalies in multivariate time-series data using a Graph Autoencoder. It identifies correlated feature anomalies and provides actionable recommendations.

Key Features:

Detects anomalies across multiple correlated process variables.

Identifies top contributing features using reconstruction error.

Provides real-time interactive dashboard with gauge visualization.

Offers actionable recommendations based on historical anomaly patterns.
"""
)

# --- Streamlit UI ---
st.title("Industrial Process Anomaly Detection & Recommendation System 🏭")
st.write("Enter the current sensor readings to get a real-time health check of the system.")

st.header("System Health Check")

input_data = {}
cols = st.columns(4)
for i, feature in enumerate(feature_names):
    min_val, max_val = normal_ranges.get(feature, (0, 100))
    help_text = f"Typical range: {min_val:.2f} to {max_val:.2f}"
    with cols[i % 4]:
        input_data[feature] = st.number_input(
            f"{feature}",
            value=float((min_val + max_val) / 2),
            format="%.2f",
            help=help_text
        )

if st.button("Analyze System Status", type="primary"):
    input_df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(input_df)
    x_input = torch.tensor(scaled_data, dtype=torch.float)
    single_edge_index = build_graph(scaled_data)
    
    with torch.no_grad():
        reconstructed, _ = model(x_input, single_edge_index)
    
    per_feature_error = (x_input - reconstructed) ** 2
    overall_score = torch.mean(per_feature_error, dim=1).item()
    normalized_score = (np.clip(overall_score / (threshold * 2), 0, 1) * 100)

    st.header("Diagnostics Report")
    
    errors = per_feature_error[0].numpy()
    indices = np.argsort(errors)[::-1]

    top_feature_1 = feature_names[indices[0]] if len(indices) > 0 else "N/A"
    top_feature_2 = feature_names[indices[1]] if len(indices) > 1 else "N/A"
    top_features = [top_feature_1, top_feature_2]

    if overall_score > threshold:
        st.error("🚨 System Status: ANOMALY DETECTED", icon="🚨")
        recommendation = random.choice(
            recommendation_rules.get(tuple(sorted(top_features)), recommendation_rules["default"])
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Abnormality Score", value=f"{normalized_score:.2f}%")
            st.markdown("##### Top Contributing Features:")
            st.info(f"1. **{top_feature_1}**")
            st.info(f"2. **{top_feature_2}**")
        with col2:
            st.warning(f"**Recommended Action:**\n\n{recommendation}")
    else:
        st.success("✅ System Status: NORMAL", icon="✅")
        st.metric(label="Abnormality Score", value=f"{normalized_score:.2f}%")
        st.info("System is operating within normal parameters. Continue to monitor.")

    # --- Speedometer / Gauge ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=normalized_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Anomaly Score (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': normalized_score
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
