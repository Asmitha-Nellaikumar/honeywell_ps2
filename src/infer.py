# src/infer.py (Final Version with Rule-Based Recommendations)
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import os
import random # Import the random library

# --- Import your custom modules ---
from src.model import GraphAutoencoder
from src.build_graph import build_graph

def run_inference(model_path, train_data_path, test_data_path):
    # --- 1. Define Paths (using passed arguments) ---
    original_data_path = r"C:\Users\NELLAIKUMAR\Desktop\analysis\data\processed\processed_honeywell_data.csv"
    output_report_path = "outputs/final_anomaly_report_rules.csv"

    # --- 2. Load Data and Model ---
    try:
        train_df = pd.read_csv(train_data_path)
        train_data = train_df.values
        test_df = pd.read_csv(test_data_path)
        test_data = test_df.values
        feature_names = train_df.columns.tolist()
    except FileNotFoundError:
        print("Error: Processed data file not found. Run the data splitting script first.")
        exit()

    num_features = train_data.shape[1]
    edge_index = build_graph(train_data)
    
    # --- KEY CHANGE: Instantiate GraphAutoencoder with positional arguments ---
    model = GraphAutoencoder(num_features) 
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- 3. Get Scores and Per-Feature Errors ---
    x_train = torch.tensor(train_data, dtype=torch.float)
    with torch.no_grad():
        reconstructed_train, _ = model(x_train, edge_index)
    train_scores = torch.mean((x_train - reconstructed_train) ** 2, dim=1).numpy()
    threshold = np.percentile(train_scores, 95)

    x_test = torch.tensor(test_data, dtype=torch.float)
    with torch.no_grad():
        reconstructed_test, _ = model(x_test, edge_index)
    per_feature_error = (x_test - reconstructed_test) ** 2
    test_scores = torch.mean(per_feature_error, dim=1).numpy()

    print("✅ Inference complete. Generating rule-based recommendations...")

    # --- 4. Create Final DataFrame ---
    original_df = pd.read_csv(original_data_path)
    results_df = original_df.iloc[len(train_data):].reset_index(drop=True)
    results_df['abnormality_score'] = (np.clip(test_scores / (threshold * 2), 0, 1) * 100).round(2)

    # --- 5. RULE-BASED RECOMMENDATION LOGIC ---
    recommendation_rules = {
        # Combination 1
        tuple(sorted(("ReactorPressurekPagauge", "ReactorTemperatureDegC"))): [
            "ACTION: Verify cooling system is active and check for outlet blockages.",
            "ACTION: Reduce reactor feed rate to lower reaction intensity.",
            "ACTION: Check the operational trend of the top contributing feature for irregularities.",
            "ACTION: Prepare for potential controlled shutdown if pressure continues to rise.",
            "ACTION: Check for leaks in the pressure relief valve."
        ],
        # Combination 2
        tuple(sorted(("CompressorWorkkW", "ReactorPressurekPagauge"))): [
            "ACTION: Inspect compressor for signs of surge or mechanical fouling.",
            "ACTION: Check recycle valve position and functionality.",
            "ACTION: Monitor compressor bearing temperatures for overheating.",
            "ACTION: Analyze gas composition for unexpected changes.",
            "ACTION: Schedule compressor for inspection during next maintenance window."
        ],
        # Default for any other single feature
        "default": [
            "ACTION: General anomaly detected. Operator to review all process variables.",
            "ACTION: Cross-reference with historical data for similar events.",
            "ACTION: Inform shift supervisor of the deviation."
        ]
    }

    recommendations = []
    for i in range(len(test_data)):
        recommendation_text = "System Normal"
        if test_scores[i] > threshold:
            errors = per_feature_error[i].numpy()
            indices = np.argsort(errors)[::-1]
            top_2_features = tuple(sorted([feature_names[j] for j in indices[:2]]))
            
            # Check if the specific 2-feature combination exists in our playbook
            if top_2_features in recommendation_rules:
                recommendation_text = random.choice(recommendation_rules[top_2_features])
            else:
                # If not, use the default recommendation
                recommendation_text = random.choice(recommendation_rules["default"])
                
        recommendations.append(recommendation_text)

    results_df['Recommended_Action'] = recommendations

    # --- 6. Save and Display Final Results ---
    print("\n--- Anomalies Report with Rule-Based Recommendations ---")
    anomalous_report_preview = results_df[results_df['abnormality_score'] > 50]
    print(anomalous_report_preview[['Time', 'abnormality_score', 'Recommended_Action']].head(10))

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv(output_report_path, index=False)
    print(f"\n✅ Full report with recommendations saved to '{output_report_path}'")
