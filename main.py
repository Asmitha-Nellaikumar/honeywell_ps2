# main.py
import os
import pandas as pd
from split_data import process_and_split_data
from src.train import train_model # KEY CHANGE: Import the correct function name
from src.infer import run_inference

def main():
    """
    Main function to run the entire anomaly detection pipeline.
    """
    print("🚀 Starting the Anomaly Detection Pipeline...")

    # Define paths
    raw_data_path = r"C:\Users\NELLAIKUMAR\Desktop\analysis\data\processed\processed_honeywell_data.csv"
    train_output_path = r"C:\Users\NELLAIKUMAR\Desktop\analysis\data\processed\train.csv"
    test_output_path = r"C:\Users\NELLAIKUMAR\Desktop\analysis\data\processed\test.csv"
    model_output_path = "outputs/models/graph_autoencoder.pt"

    # --- Step 1: Process and Split Data ---
    print("\n[Step 1/3] Processing and splitting data...")
    process_and_split_data(raw_data_path, train_output_path, test_output_path)
    print("✅ Data processing complete!")

    # --- Step 2: Train the Model ---
    print("\n[Step 2/3] Training the model...")
    train_df = pd.read_csv(train_output_path)
    train_data = train_df.values
    num_features = train_data.shape[1]
    
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # KEY CHANGE: Call the new function name and pass the correct arguments
    train_model(train_data, num_features, model_save_path=model_output_path)
    print("✅ Model training complete!")

    # --- Step 3: Run Inference and Get Results ---
    print("\n[Step 3/3] Running inference to detect anomalies...")
    run_inference(model_output_path, train_output_path, test_output_path)
    print("\n✅ Pipeline finished successfully! Check the plots for results.")

if __name__ == "__main__":
    main()
