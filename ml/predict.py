import pandas as pd
import numpy as np
import logging
import joblib
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Load a trained model from disk.

    Args:
        model_path (str): Path to the saved model

    Returns:
        object: Loaded model
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        # Load model
        model = joblib.load(model_path)

        logger.info(f"Loaded model from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def preprocess_input(data, preprocessor):
    """
    Preprocess input data for prediction.

    Args:
        data (dict or pandas.DataFrame): Input data
        preprocessor: Fitted preprocessor object

    Returns:
        numpy.ndarray: Preprocessed input data
    """
    try:
        # Convert input to DataFrame if it's a dictionary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Input data must be a dictionary or DataFrame")

        # Preprocess data
        processed_data = preprocessor.transform(df)

        return processed_data

    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise


def make_prediction(model, preprocessed_data):
    """
    Make prediction using trained model.

    Args:
        model: Trained model
        preprocessed_data: Preprocessed input data

    Returns:
        dict: Prediction results
    """
    try:
        # Make prediction
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)

        # Return results
        result = {
            "prediction": prediction.tolist(),
            "probability": prediction_proba.tolist(),
        }

        return result

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def preprocess_and_predict(data, model_path, preprocessor_path=None):
    print("MODEL PATH", model_path)
    """
    Preprocess input data and make prediction.

    Args:
        data (dict or pandas.DataFrame): Input data
        model_path (str): Path to the saved model
        preprocessor_path (str, optional): Path to the saved preprocessor. If None, assumes it's in the same directory with '_preprocessor' suffix.

    Returns:
        dict: Prediction results
    """
    try:
        # Determine preprocessor path if not provided
        if preprocessor_path is None:
            preprocessor_path = model_path.replace(
                "xgboost_credit_risk", "preprocessor"
            )

        # Load model and preprocessor
        model = load_model(model_path)
        if model is None:
            raise ValueError(f"Failed to load model from {model_path}")

        from ml.preprocess import load_preprocessor

        preprocessor = load_preprocessor(preprocessor_path)
        if preprocessor is None:
            raise ValueError(f"Failed to load preprocessor from {preprocessor_path}")

        # Preprocess input data
        preprocessed_data = preprocess_input(data, preprocessor)

        # Make prediction
        result = make_prediction(model, preprocessed_data)

        # Add loan status interpretation
        result["status_interpretation"] = []
        for pred in result["prediction"]:
            if pred == 1:
                result["status_interpretation"].append("Default")
            else:
                result["status_interpretation"].append("Non-Default")

        # Add probability of default (taking the probability of class 1)
        result["default_probability"] = [prob[1] for prob in result["probability"]]

        return result

    except Exception as e:
        logger.error(f"Error in preprocess_and_predict: {str(e)}")
        raise
