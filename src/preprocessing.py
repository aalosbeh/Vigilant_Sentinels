import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    print(f"[INFO] Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df):
    print("[INFO] Cleaning data: Removing duplicates and filling missing values...")
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    print("[INFO] Data cleaning complete.")
    return df

def encode_categorical(df, column):
    print(f"[INFO] Encoding categorical column: {column}...")
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    print("[INFO] Encoding complete.")
    return df

def scale_features(df, columns):
    print(f"[INFO] Scaling numeric features: {columns}...")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print("[INFO] Scaling complete.")
    return df
