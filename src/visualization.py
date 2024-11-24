import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, save_path):
    print("[INFO] Generating Predicted vs. Actual values plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.savefig(save_path)
    print(f"[INFO] Predicted vs Actual values plot saved to {save_path}.")
    plt.show()

def plot_residuals(y_test, y_pred, save_path):
    print("[INFO] Generating residual plot...")
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.savefig(save_path)
    print(f"[INFO] Residual plot saved to {save_path}.")
    plt.show()