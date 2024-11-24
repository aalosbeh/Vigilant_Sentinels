import matplotlib.pyplot as plt
import seaborn as sns

def describe_dataset(df):
    print("[INFO] Dataset summary statistics:")
    print(df.describe())
    print("\n[INFO] Missing Values:")
    print(df.isnull().sum())

def plot_correlations(df, save_path):
    print("[INFO] Generating correlation heatmap...")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(save_path)
    print(f"[INFO] Correlation heatmap saved to {save_path}.")
    plt.show()