# Credit Risk Analytics - Backend API

The backend component of the Credit Risk Analytics platform provides a Flask-based REST API for processing, analyzing, and serving credit risk data, along with machine learning capabilities for risk prediction.

## Quick Start

```bash
# Clone repository (if applicable)
git clone <repository-url>
cd credit-risk-analytics/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python setup_db.py
# OR
export FLASK_APP=app.py
flask init-db

# Run ETL pipeline
flask etl-pipeline

# Train ML model
flask train-model

# Run the application
flask run
# OR
python app.py
```

## Project Structure

```
backend/
├── app.py                  # Main Flask application
├── config/                 # Configuration settings
│   └── config.py
├── commands.py             # Flask CLI commands
├── data/                   # Data files
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── etl/                    # ETL pipeline components
│   ├── extract.py          # Data extraction functions
│   ├── transform.py        # Data transformation functions
│   └── load.py             # Data loading functions
├── models/                 # Database models
│   └── credit_risk.py      # Credit risk model definitions
├── ml/                     # Machine learning components
│   ├── preprocess.py       # Feature preprocessing
│   ├── train.py            # Model training
│   └── predict.py          # Model prediction
├── templates/              # HTML templates
│   ├── admin.html          # Admin dashboard
│   └── error.html          # Error page
└── README.md               # This file
```

## API Endpoints

### Admin and Dashboard

- `GET /admin` - Admin dashboard interface
- `GET /` - Redirects to admin dashboard

### API Health

- `GET /api/health` - API health check endpoint

### Data Endpoints

- `GET /api/data/summary` - Get summary statistics of credit risk data
- `GET /api/data` - Get paginated credit risk data
  - Query parameters:
    - `page` (int): Page number (default: 1)
    - `per_page` (int): Records per page (default: 100)
    - `loan_status` (int): Filter by loan status (0 = non-default, 1 = default)

Example response:

```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "person_age": 30,
      "person_income": 60000,
      "loan_status": 0,
      "...": "..."
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 100,
    "total_pages": 10,
    "total_records": 1000
  }
}
```

### Machine Learning Endpoints

- `POST /api/ml/predict` - Make credit risk prediction
  - Request body: JSON with loan applicant data
  - Response: Prediction result with confidence score

Example request:

```json
{
  "person_age": 35,
  "person_income": 85000,
  "person_home_ownership": "MORTGAGE",
  "person_emp_length": 12.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 15000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.18,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 15
}
```

Example response:

```json
{
  "status": "success",
  "prediction": [0],
  "probability": [[0.85, 0.15]],
  "status_interpretation": ["Non-Default"],
  "default_probability": [0.15],
  "model_info": {
    "id": 1,
    "name": "xgboost_credit_risk",
    "version": "v1_20230401_123456",
    "metrics": "..."
  }
}
```

## CLI Commands

The backend provides several Flask CLI commands:

- `flask init-db` - Initialize the database
- `flask load-data` - Load data from CSV into the database
- `flask etl-pipeline` - Run the full ETL pipeline
- `flask train-model` - Train a machine learning model
- `flask data-summary` - Print a summary of the data in the database

## Configuration

The backend uses a configuration class system:

- `Config` - Base configuration class
- `DevelopmentConfig` - Development environment (default)
- `TestingConfig` - Testing environment
- `ProductionConfig` - Production environment

Environment variables:

- `FLASK_APP`: Set to `app.py`
- `FLASK_ENV`: Set to `development`, `testing`, or `production`
- `DATABASE_URL`: Database connection string for production
- `SECRET_KEY`: Secret key for secure sessions

## Dependencies

Main dependencies:

- Flask - Web framework
- SQLAlchemy - ORM
- Pandas - Data processing
- Scikit-learn - Machine learning
- XGBoost - Gradient boosting

See `requirements.txt` for the complete list.

## Data Source

This project uses the [Credit Risk Dataset](https://www.kaggle.com/laotse/credit-risk-dataset) from Kaggle. The raw CSV file should be placed in the `data/raw/` directory.
