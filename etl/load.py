import os
import pandas as pd
import logging
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_to_db(df, db, batch_size=1000):
    """
    Load transformed data into the database.

    Args:
        df (pandas.DataFrame): Transformed DataFrame to load
        db: SQLAlchemy database instance
        batch_size (int): Number of records to insert in each batch

    Returns:
        bool: True if successful, False otherwise
    """
    from models.credit_risk import CreditRisk

    try:
        logger.info(f"Loading {len(df)} records to database")

        # Convert DataFrame to dictionary and bulk insert in batches
        records = df.to_dict(orient="records")
        total_records = len(records)

        # Process in batches to avoid memory issues with large datasets
        for i in range(0, total_records, batch_size):
            batch = records[i : min(i + batch_size, total_records)]
            db.session.bulk_insert_mappings(CreditRisk, batch)
            db.session.commit()
            logger.info(
                f"Inserted batch {i//batch_size + 1}/{(total_records+batch_size-1)//batch_size}"
            )

        logger.info(f"Successfully loaded {total_records} records to database")
        return True

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database error: {str(e)}")
        return False

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error loading data to database: {str(e)}")
        return False


def save_to_csv(df, output_path):
    """
    Save transformed data to CSV file.

    Args:
        df (pandas.DataFrame): Transformed DataFrame to save
        output_path (str): Path to save the CSV file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Successfully saved {len(df)} records to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving data to CSV: {str(e)}")
        return False


def update_data_in_db(df, db, id_column="id"):
    """
    Update existing records in the database.

    Args:
        df (pandas.DataFrame): DataFrame with updated records
        db: SQLAlchemy database instance
        id_column (str): Name of the ID column for matching records

    Returns:
        bool: True if successful, False otherwise
    """
    from models.credit_risk import CreditRisk

    try:
        logger.info(f"Updating {len(df)} records in database")

        # Ensure ID column exists
        if id_column not in df.columns:
            logger.error(f"ID column {id_column} not found in DataFrame")
            return False

        # Convert DataFrame to dictionary and bulk update
        records = df.to_dict(orient="records")
        db.session.bulk_update_mappings(CreditRisk, records)
        db.session.commit()

        logger.info(f"Successfully updated {len(df)} records in database")
        return True

    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database error: {str(e)}")
        return False

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating data in database: {str(e)}")
        return False
