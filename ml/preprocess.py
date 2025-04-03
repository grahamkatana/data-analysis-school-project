# import pandas as pd
# import numpy as np
# import logging
# import joblib
# import os
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# def create_preprocessing_pipeline(categorical_features=None, numeric_features=None):
#     """
#     Create a preprocessing pipeline for categorical and numeric features.
#     """
#     if categorical_features is None:
#         categorical_features = [
#             "person_home_ownership",
#             "loan_intent",
#             "loan_grade",
#             "cb_person_default_on_file",
#         ]

#     if numeric_features is None:
#         numeric_features = [
#             "person_age",
#             "person_income",
#             "person_emp_length",
#             "loan_amnt",
#             "loan_int_rate",
#             "loan_percent_income",
#             "cb_person_cred_hist_length",
#             "debt_to_income",
#             "income_to_loan_ratio",
#             "credit_history_years_per_age",
#             "employment_length_to_age_ratio",
#             "loan_to_income_percent",
#         ]

#     numeric_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#         ]
#     )

#     categorical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features),
#         ],
#         remainder="drop",
#     )

#     return preprocessor


# def preprocess_data(df, target_column="loan_status", test_size=0.2, random_state=42):
#     """
#     Preprocess data for machine learning.
#     """
#     from sklearn.model_selection import train_test_split

#     logger.info("Preprocessing data for machine learning")

#     df_copy = df.copy()

#     if target_column not in df_copy.columns:
#         raise ValueError(f"Target column '{target_column}' not found in DataFrame")

#     X = df_copy.drop(columns=[target_column])
#     y = df_copy[target_column]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )

#     logger.info(
#         f"Split data into train ({len(X_train)} rows) and test ({len(X_test)} rows) sets"
#     )

#     # Correct Feature Type Separation
#     categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
#     numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

#     preprocessor = create_preprocessing_pipeline(
#         categorical_features=categorical_features, numeric_features=numeric_features
#     )

#     logger.info("Fitting preprocessing pipeline on training data")
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)

#     categorical_ohe_feature_names = []
#     if categorical_features:
#         ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
#         categorical_ohe_feature_names = ohe.get_feature_names_out(
#             categorical_features
#         ).tolist()

#     feature_names = numeric_features + categorical_ohe_feature_names

#     logger.info(
#         f"Preprocessing complete. X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}"
#     )

#     return {
#         "X_train": X_train_processed,
#         "X_test": X_test_processed,
#         "y_train": y_train,
#         "y_test": y_test,
#         "preprocessor": preprocessor,
#         "feature_names": feature_names,
#         "categorical_features": categorical_features,
#         "numeric_features": numeric_features,
#     }


# def save_preprocessor(preprocessor, preprocessor_path):
#     """
#     Save the preprocessing pipeline to disk.

#     Args:
#         preprocessor: Fitted preprocessor object
#         preprocessor_path (str): Path to save the preprocessor

#     Returns:
#         bool: True if successful, False otherwise
#     """
#     try:
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

#         # Save preprocessor
#         joblib.dump(preprocessor, preprocessor_path)

#         logger.info(f"Saved preprocessor to {preprocessor_path}")
#         return True

#     except Exception as e:
#         logger.error(f"Error saving preprocessor: {str(e)}")
#         return False


# def load_preprocessor(preprocessor_path):
#     """
#     Load preprocessing pipeline from disk.

#     Args:
#         preprocessor_path (str): Path to the saved preprocessor

#     Returns:
#         object: Loaded preprocessor
#     """
#     try:
#         # Check if file exists
#         if not os.path.exists(preprocessor_path):
#             logger.error(f"Preprocessor file not found: {preprocessor_path}")
#             return None

#         # Load preprocessor
#         preprocessor = joblib.load(preprocessor_path)

#         logger.info(f"Loaded preprocessor from {preprocessor_path}")
#         return preprocessor

#     except Exception as e:
#         logger.error(f"Error loading preprocessor: {str(e)}")
#         return None


import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(categorical_features=None, numeric_features=None):
    if categorical_features is None:
        categorical_features = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]

    if numeric_features is None:
        numeric_features = [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "debt_to_income",
            "income_to_loan_ratio",
            "credit_history_years_per_age",
            "employment_length_to_age_ratio",
            "loan_to_income_percent",
        ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess_data(df, target_column="loan_status", test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    logger.info("Preprocessing data for machine learning")

    df_copy = df.copy()

    if target_column not in df_copy.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Split data into train ({len(X_train)} rows) and test ({len(X_test)} rows) sets"
    )

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = create_preprocessing_pipeline(
        categorical_features=categorical_features, numeric_features=numeric_features
    )

    logger.info("Fitting preprocessing pipeline on training data")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    categorical_ohe_feature_names = []
    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        categorical_ohe_feature_names = ohe.get_feature_names_out(
            categorical_features
        ).tolist()

    feature_names = numeric_features + categorical_ohe_feature_names

    logger.info(
        f"Preprocessing complete. X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}"
    )

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }


def save_preprocessor(preprocessor, preprocessor_path):
    try:
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    except Exception as e:
        logger.error(f"Error saving preprocessor: {str(e)}")
        raise


def load_preprocessor(preprocessor_path):
    try:
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor file not found: {preprocessor_path}")
            return None

        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
        return preprocessor

    except Exception as e:
        logger.error(f"Error loading preprocessor: {str(e)}")
        return None
