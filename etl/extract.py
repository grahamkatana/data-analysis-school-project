# backend/etl/extract.py
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_data(file_path=None, sample_size=None):
    """
    Extract data from the credit risk dataset CSV file.

    Args:
        file_path (str): Path to the CSV file. If None, uses default path.
        sample_size (int, optional): Number of records to sample. If None, uses all data.

    Returns:
        pandas.DataFrame: The extracted data
    """
    try:
        # Default path if none provided
        if file_path is None:
            file_path = "data/raw/credit_risk_dataset.csv"

        logger.info(f"Extracting data from {file_path}")

        # Check if file exists
        import os

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Log the shape of the dataset
        logger.info(f"Dataset shape: {df.shape}")

        # Sample data if requested
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records from dataset")

        # Verify data was loaded successfully
        if df.empty:
            logger.warning("Extracted DataFrame is empty!")
        else:
            logger.info(
                f"Successfully extracted {len(df)} records with {df.shape[1]} features"
            )

        return df

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
