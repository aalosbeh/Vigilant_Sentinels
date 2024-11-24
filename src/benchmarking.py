def benchmark_models(df, performance_metrics, time_metrics):
    """Rank models based on performance and efficiency."""
    df["performance_score"] = df[performance_metrics].mean(axis=1)
    df["efficiency_score"] = 1 / (df[time_metrics].sum(axis=1))
    df["overall_score"] = df["performance_score"] * df["efficiency_score"]
    return df.sort_values(by="overall_score", ascending=False)

def compare_models(df, model_column):
    """Compare model performance."""
    return df.groupby(model_column).mean()
