import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model_pipeline(input_file, model_type="xgboost", model_dir=None):
    """
    Run the full model training pipeline.

    Args:
        input_file (str): Path to input CSV file
        model_type (str): Type of model to train ('xgboost', 'random_forest', 'logistic')
        model_dir (str): Directory to save model files

    Returns:
        dict: Dictionary containing model metadata
    """
    from ml.preprocess import preprocess_data, save_preprocessor

    try:
        logger.info(f"Starting model training pipeline for model type: {model_type}")

        # Default model directory if not provided
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ml",
                "models",
            )

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")

        # Preprocess data
        processed_data = preprocess_data(df)

        # Apply SMOTE for handling imbalanced data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            processed_data["X_train"], processed_data["y_train"]
        )
        logger.info(
            f"Applied SMOTE. Original train shape: {processed_data['X_train'].shape}, Resampled shape: {X_train_resampled.shape}"
        )

        # Train model based on model type
        if model_type == "xgboost":
            model, best_params = train_xgboost(X_train_resampled, y_train_resampled)
        elif model_type == "random_forest":
            model, best_params = train_random_forest(
                X_train_resampled, y_train_resampled
            )
        elif model_type == "logistic":
            model, best_params = train_logistic_regression(
                X_train_resampled, y_train_resampled
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Evaluate model
        metrics = evaluate_model(
            model, processed_data["X_test"], processed_data["y_test"]
        )

        # Generate model info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_credit_risk"
        model_version = f"v1_{timestamp}"
        model_path = os.path.join(model_dir, f"{model_name}_{model_version}.joblib")
        preprocessor_path = os.path.join(
            model_dir, f"preprocessor_{model_version}.joblib"
        )

        # Save model and preprocessor
        joblib.dump(model, model_path)
        save_preprocessor(processed_data["preprocessor"], preprocessor_path)

        # Save model metadata
        metadata = {
            "name": model_name,
            "version": model_version,
            "model_type": model_type,
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "metrics": metrics,
            "parameters": best_params,
            "feature_names": processed_data["feature_names"],
            "categorical_features": processed_data["categorical_features"],
            "numeric_features": processed_data["numeric_features"],
            "trained_at": timestamp,
        }

        # Save metadata to JSON
        metadata_path = os.path.join(
            model_dir, f"{model_name}_{model_version}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Model training pipeline completed. Model saved to {model_path}")
        return metadata

    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise


def train_xgboost(X_train, y_train):
    """Trains an XGBoost model."""
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }
    xgb_model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=StratifiedKFold(n_splits=3),
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_


def train_random_forest(X_train, y_train):
    """Trains a Random Forest model."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=StratifiedKFold(n_splits=3),
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_


def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    lr_model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        lr_model,
        param_grid,
        cv=StratifiedKFold(n_splits=3),
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics
