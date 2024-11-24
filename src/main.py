import sys
import os
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, PLOT_DIR, TARGET_COLUMN
from src.preprocessing import load_data, clean_data, encode_categorical, scale_features
from src.feature_engineering import add_efficiency_metrics, create_composite_score
from src.model_training import train_sklearn_model, train_bert
import pandas as pd

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("PYTHONPATH:", sys.path)

def benchmark_models(df):
    models = ["RandomForest", "GradientBoosting", "LinearRegression", "SVM"]
    benchmark_results = []

    for model_type in models:
        print(f"=== Training {model_type} ===")
        _, _, _, metrics = train_sklearn_model(df, TARGET_COLUMN, model_type=model_type)
        metrics["model"] = model_type
        benchmark_results.append(metrics)

    print("=== Training BERT ===")
    _, _, _, metrics = train_bert(df, TARGET_COLUMN)
    metrics["model"] = "BERT"
    benchmark_results.append(metrics)

    return pd.DataFrame(benchmark_results)


def main():
    print("=== Step 1: Loading and Preprocessing Data ===")
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = encode_categorical(df, "Model")
    df = scale_features(df, ["f1", "precision", "recall", "accuracy", "training_time", "inference_time"])
    df = add_efficiency_metrics(df)
    df = create_composite_score(df)

    print("=== Step 2: Benchmarking Models ===")
    benchmark_df = benchmark_models(df)
    benchmark_path = os.path.join(PLOT_DIR, "benchmark_results.csv")
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"Benchmark results saved to {benchmark_path}")


if __name__ == "__main__":
    main()