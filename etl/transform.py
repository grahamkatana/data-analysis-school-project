import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transform_data(df):
    """
    Transform and clean the credit risk dataset.

    Args:
        df (pandas.DataFrame): Raw DataFrame from extract step

    Returns:
        pandas.DataFrame: Cleaned and transformed DataFrame
    """
    logger.info("Starting data transformation")

    # Make a copy to avoid modifying the original
    df_transformed = df.copy()

    # 1. Check for missing values
    missing_values = df_transformed.isnull().sum()
    logger.info(f"Missing values before handling: {missing_values[missing_values > 0]}")

    # 2. Handle missing values
    df_transformed = handle_missing_values(df_transformed)

    # 3. Convert datatypes
    df_transformed = convert_datatypes(df_transformed)

    # 4. Feature encoding
    df_transformed = encode_categorical_features(df_transformed)

    # 5. Feature engineering
    df_transformed = engineer_features(df_transformed)

    # 6. Handle outliers
    df_transformed = handle_outliers(df_transformed)

    # 7. Data validation
    df_transformed = validate_data(df_transformed)

    # Log transformation summary
    logger.info(
        f"Transformation complete. Input shape: {df.shape}, Output shape: {df_transformed.shape}"
    )

    return df_transformed


def handle_missing_values(df):
    """
    Handle missing values in the dataset.

    Args:
        df (pandas.DataFrame): DataFrame with missing values

    Returns:
        pandas.DataFrame: DataFrame with handled missing values
    """
    logger.info("Handling missing values")

    # Make a copy
    df_cleaned = df.copy()

    # Log missing value counts
    missing_counts = df_cleaned.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Missing values found: {missing_counts[missing_counts > 0]}")

    # Handle numeric columns with mean imputation
    numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            logger.info(f"Imputed {col} with median value: {median_value}")

    # Handle categorical columns with mode imputation
    categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode_value, inplace=True)
            logger.info(f"Imputed {col} with mode value: {mode_value}")

    # Verify no missing values remain
    assert (
        df_cleaned.isnull().sum().sum() == 0
    ), "Missing values still exist after imputation"

    return df_cleaned


def convert_datatypes(df):
    """
    Convert columns to appropriate datatypes.

    Args:
        df (pandas.DataFrame): DataFrame with potential datatype issues

    Returns:
        pandas.DataFrame: DataFrame with corrected datatypes
    """
    logger.info("Converting datatypes")

    # Make a copy
    df_converted = df.copy()

    # Define datatype conversions
    int_columns = [
        "person_age",
        "person_income",
        "loan_amnt",
        "cb_person_cred_hist_length",
    ]
    float_columns = ["person_emp_length", "loan_int_rate", "loan_percent_income"]

    # Convert to integers
    for col in int_columns:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].astype(int)

    # Convert to floats
    for col in float_columns:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].astype(float)

    return df_converted


def encode_categorical_features(df):
    """
    Encode categorical features for machine learning compatibility.

    Args:
        df (pandas.DataFrame): DataFrame with categorical features

    Returns:
        pandas.DataFrame: DataFrame with encoded categorical features
    """
    logger.info("Encoding categorical features")

    # Make a copy
    df_encoded = df.copy()

    # Identify categorical columns
    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    # Apply label encoding
    label_encoders = {}
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            logger.info(f"Encoded {col} with {len(le.classes_)} unique values")

    return df_encoded


def engineer_features(df):
    """
    Create new features from existing data.

    Args:
        df (pandas.DataFrame): DataFrame to engineer features for

    Returns:
        pandas.DataFrame: DataFrame with new engineered features
    """
    logger.info("Engineering new features")

    # Make a copy
    df_engineered = df.copy()

    # Create debt to income ratio
    if (
        "person_income" in df_engineered.columns
        and "loan_amnt" in df_engineered.columns
    ):
        df_engineered["debt_to_income"] = (
            df_engineered["loan_amnt"] / df_engineered["person_income"]
        )
        logger.info("Created 'debt_to_income' feature")

    # Create income to loan ratio
    if (
        "person_income" in df_engineered.columns
        and "loan_amnt" in df_engineered.columns
    ):
        df_engineered["income_to_loan_ratio"] = (
            df_engineered["person_income"] / df_engineered["loan_amnt"]
        )
        logger.info("Created 'income_to_loan_ratio' feature")

    # Create credit history years per age
    if (
        "person_age" in df_engineered.columns
        and "cb_person_cred_hist_length" in df_engineered.columns
    ):
        df_engineered["credit_history_years_per_age"] = (
            df_engineered["cb_person_cred_hist_length"] / df_engineered["person_age"]
        )
        logger.info("Created 'credit_history_years_per_age' feature")

    # Create employment length to age ratio
    if (
        "person_age" in df_engineered.columns
        and "person_emp_length" in df_engineered.columns
    ):
        df_engineered["employment_length_to_age_ratio"] = (
            df_engineered["person_emp_length"] / df_engineered["person_age"]
        )
        logger.info("Created 'employment_length_to_age_ratio' feature")

    # Create loan amount to income percentage (different from existing loan_percent_income)
    if (
        "person_income" in df_engineered.columns
        and "loan_amnt" in df_engineered.columns
    ):
        df_engineered["loan_to_income_percent"] = (
            df_engineered["loan_amnt"] / df_engineered["person_income"]
        ) * 100
        logger.info("Created 'loan_to_income_percent' feature")

    return df_engineered


def handle_outliers(df, cols_to_check=None, method="iqr", threshold=1.5):
    """
    Handle outliers in the dataset.

    Args:
        df (pandas.DataFrame): DataFrame to handle outliers in
        cols_to_check (list): List of columns to check for outliers
        method (str): Method to use for outlier detection ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection

    Returns:
        pandas.DataFrame: DataFrame with handled outliers
    """
    logger.info(f"Handling outliers using {method} method with threshold {threshold}")

    # Make a copy
    df_no_outliers = df.copy()

    # Default columns to check if not specified
    if cols_to_check is None:
        cols_to_check = [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
        ]

    # Ensure all specified columns exist in the DataFrame
    cols_to_check = [col for col in cols_to_check if col in df_no_outliers.columns]

    # Count total rows before outlier removal
    total_rows_before = len(df_no_outliers)

    # IQR method
    if method == "iqr":
        for col in cols_to_check:
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Cap outliers instead of removing them
            df_no_outliers[col] = df_no_outliers[col].clip(
                lower=lower_bound, upper=upper_bound
            )
            logger.info(
                f"Capped outliers in {col}: < {lower_bound:.2f} or > {upper_bound:.2f}"
            )

    # Z-score method
    elif method == "zscore":
        for col in cols_to_check:
            z_scores = np.abs(stats.zscore(df_no_outliers[col]))
            outliers = z_scores > threshold

            # Cap outliers using the maximum non-outlier value
            if sum(outliers) > 0:
                non_outlier_max = df_no_outliers.loc[~outliers, col].max()
                non_outlier_min = df_no_outliers.loc[~outliers, col].min()

                # Cap upper outliers
                upper_outliers = df_no_outliers[col] > non_outlier_max
                df_no_outliers.loc[upper_outliers, col] = non_outlier_max

                # Cap lower outliers
                lower_outliers = df_no_outliers[col] < non_outlier_min
                df_no_outliers.loc[lower_outliers, col] = non_outlier_min

                logger.info(
                    f"Capped {sum(outliers)} outliers in {col} using z-score method"
                )

    logger.info(
        f"Outlier handling complete. Total rows: {total_rows_before} (maintained all rows by capping)"
    )
    return df_no_outliers


def validate_data(df):
    """
    Validate data after all transformations.

    Args:
        df (pandas.DataFrame): Transformed DataFrame to validate

    Returns:
        pandas.DataFrame: Validated DataFrame
    """
    logger.info("Validating transformed data")

    # Make a copy
    df_validated = df.copy()

    # Check for any remaining missing values
    missing_values = df_validated.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(
            f"Missing values after transformation: {missing_values[missing_values > 0]}"
        )
        # Fill any remaining missing values
        df_validated = df_validated.fillna(0)

    # Validate data types
    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]

    for col in numeric_cols:
        if col in df_validated.columns:
            if not pd.api.types.is_numeric_dtype(df_validated[col]):
                logger.warning(f"Column {col} is not numeric. Converting to numeric.")
                df_validated[col] = pd.to_numeric(df_validated[col], errors="coerce")

    # Validate value ranges
    # Age should be positive and reasonable
    if "person_age" in df_validated.columns:
        if (df_validated["person_age"] < 18).any():
            logger.warning("Found age values below 18")
            df_validated.loc[df_validated["person_age"] < 18, "person_age"] = 18

        if (df_validated["person_age"] > 100).any():
            logger.warning("Found age values above 100")
            df_validated.loc[df_validated["person_age"] > 100, "person_age"] = 100

    # Income should be positive
    if "person_income" in df_validated.columns:
        if (df_validated["person_income"] <= 0).any():
            logger.warning("Found non-positive income values")
            min_positive_income = df_validated.loc[
                df_validated["person_income"] > 0, "person_income"
            ].min()
            df_validated.loc[df_validated["person_income"] <= 0, "person_income"] = (
                min_positive_income
            )

    # Loan amount should be positive
    if "loan_amnt" in df_validated.columns:
        if (df_validated["loan_amnt"] <= 0).any():
            logger.warning("Found non-positive loan amount values")
            min_positive_loan = df_validated.loc[
                df_validated["loan_amnt"] > 0, "loan_amnt"
            ].min()
            df_validated.loc[df_validated["loan_amnt"] <= 0, "loan_amnt"] = (
                min_positive_loan
            )

    # Interest rate should be positive and reasonable
    if "loan_int_rate" in df_validated.columns:
        if (df_validated["loan_int_rate"] < 0).any():
            logger.warning("Found negative interest rate values")
            df_validated.loc[df_validated["loan_int_rate"] < 0, "loan_int_rate"] = 0

        if (df_validated["loan_int_rate"] > 100).any():
            logger.warning("Found interest rate values above 100%")
            df_validated.loc[df_validated["loan_int_rate"] > 100, "loan_int_rate"] = 100

    logger.info("Data validation complete")
    return df_validated
