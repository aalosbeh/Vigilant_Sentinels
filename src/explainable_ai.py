import shap

def explain_predictions(model, X):
    """Explain model predictions using SHAP."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)
    shap.decision_plot(explainer.expected_value[1], shap_values[1], X)
