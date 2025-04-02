import os
from datetime import timedelta

# Get the project root directory (one level up from config folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """Base configuration class"""

    # Flask config
    DEBUG = False
    TESTING = False
    SECRET_KEY = "testsupersecret@1"

    # SQLAlchemy config
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Data paths - Use BASE_DIR instead of __file__ to get correct paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # ML model paths
    MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

    # Ensure directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Default dataset filename
    DEFAULT_DATASET_FILENAME = "credit_risk_dataset.csv"
    DEFAULT_DATASET_PATH = os.path.join(RAW_DATA_DIR, DEFAULT_DATASET_FILENAME)


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DEV_DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "dev.sqlite"),
    )
