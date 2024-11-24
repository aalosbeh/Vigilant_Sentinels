import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification



def train_sklearn_model(df, target_column, model_type="RandomForest", test_size=0.2, random_state=42):
    """
    Train a traditional ML model using sklearn.
    """
    print(f"[INFO] Training {model_type} model...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Select model
    if model_type == "RandomForest":
        model = RandomForestRegressor(random_state=random_state)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=random_state)
    elif model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "SVM":
        model = SVR()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("[INFO] Model evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    return model, y_test, y_pred, {"mse": mse, "mae": mae, "r2": r2, "training_time": training_time}


def train_bert(df, target_column, test_size=0.2, random_state=42):
    """
    Train a BERT model for regression using HuggingFace Transformers.
    """
    print("[INFO] Training BERT model...")

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(example):
        return tokenizer(example, padding="max_length", truncation=True, max_length=128)

    # Split data
    texts = df.drop(columns=[target_column]).apply(lambda row: " ".join(map(str, row)), axis=1).tolist()
    targets = df[target_column].tolist()
    train_texts, test_texts, train_targets, test_targets = train_test_split(texts, targets, test_size=test_size, random_state=random_state)

    # Prepare Torch dataset
    class RegressionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, targets):
            self.encodings = encodings
            self.targets = targets

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
            return item

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    train_dataset = RegressionDataset(train_encodings, train_targets)
    test_dataset = RegressionDataset(test_encodings, test_targets)

    # Model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        eval_strategy="epoch",  # Changed from 'evaluation_strategy' to 'eval_strategy'
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Evaluation
    preds = trainer.predict(test_dataset)
    mse = mean_squared_error(test_targets, preds.predictions.flatten())
    mae = mean_absolute_error(test_targets, preds.predictions.flatten())
    r2 = r2_score(test_targets, preds.predictions.flatten())

    print("[INFO] BERT Model evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    return model, test_targets, preds.predictions.flatten(), {"mse": mse, "mae": mae, "r2": r2, "training_time": training_time}