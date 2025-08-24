# split_data.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

def process_and_split_data(raw_data_path, train_output_path, test_output_path, scaler_path="outputs/models/scaler.joblib"):
    """
    Loads raw data, scales features, splits into train/test sets, and saves the
    processed data and scaler.
    """
    print("Processing and splitting data...")
    
    # --- Load Dataset ---
    df = pd.read_csv(raw_data_path)

    # Drop timestamp if present
    if "Time" in df.columns:
        features = df.drop(columns=["Time"])
    else:
        features = df

    # --- Scale Features ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Split into Train/Test ---
    split_index = int(0.6 * len(scaled_features))
    train_data = scaled_features[:split_index]
    test_data = scaled_features[split_index:]

    print(f"✅ Data processed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # --- Save ---
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    pd.DataFrame(train_data, columns=features.columns).to_csv(train_output_path, index=False)
    pd.DataFrame(test_data, columns=features.columns).to_csv(test_output_path, index=False)
    joblib.dump(scaler, scaler_path)

    print("✅ Train/Test CSV and scaler saved successfully!")
    
if __name__ == "__main__":
    # --- Define Paths ---
    RAW_DATA_PATH = r"C:\Users\NELLAIKUMAR\Desktop\analysis\data\processed\processed_honeywell_data.csv"
    TRAIN_OUTPUT_PATH = "data/processed/train.csv"
    TEST_OUTPUT_PATH = "data/processed/test.csv"
    SCALER_PATH = "outputs/models/scaler.pkl"
    
    process_and_split_data(RAW_DATA_PATH, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH, SCALER_PATH)

