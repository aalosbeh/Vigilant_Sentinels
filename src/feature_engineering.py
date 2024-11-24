def add_efficiency_metrics(df):
    print("[INFO] Adding efficiency metrics...")
    df["training_inference_ratio"] = df["training_time"] / df["inference_time"]
    print("[INFO] Efficiency metrics added.")
    return df

def create_composite_score(df):
    print("[INFO] Adding composite score...")
    df["composite_score"] = (df["f1"] + df["precision"] + df["recall"] + df["accuracy"]) / 4
    print("[INFO] Composite score added.")
    return df