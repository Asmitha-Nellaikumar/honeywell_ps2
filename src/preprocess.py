# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(path, train_ratio=0.6):
    df = pd.read_csv(r"C:\Users\NELLAIKUMAR\Desktop\analysis\data\processed\processed_honeywell_data.csv")
    
    # scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    
    # time split
    split_idx = int(len(scaled) * train_ratio)
    train, test = scaled[:split_idx], scaled[split_idx:]
    return train, test, scaler
