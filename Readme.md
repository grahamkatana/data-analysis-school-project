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
├── swagger.py              # OpenAPI/Swagger definitions
├── templates/              # HTML templates
│   ├── admin.html          # Admin dashboard
│   └── error.html          # Error page
└── README.md               # This file
```

## API Endpoints

### Admin and Dashboard

- `GET /admin` - Admin dashboard interface
- `GET /` - Redirects to admin dashboard
- `GET /api/docs` - Interactive API documentation (Swagger UI)

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
  - Request body: JSON with loan applicant data (numeric format)
  - Response: Prediction result with confidence score

Example request (using numeric categorical encodings):

```json
{
  "person_age": 35,
  "person_income": 85000,
  "person_home_ownership": 1,
  "person_emp_length": 12.0,
  "loan_intent": 3,
  "loan_grade": 1,
  "loan_amnt": 15000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.18,
  "cb_person_default_on_file": 0,
  "cb_person_cred_hist_length": 15,
  "debt_to_income": 0.32,
  "income_to_loan_ratio": 5.67
}
```

Example response:

```json
{
  "status": "success",
  "prediction": [0],
  "probability": [[0.9068, 0.0932]],
  "status_interpretation": ["Non-Default"],
  "default_probability": [0.0932],
  "model_info": {
    "id": 6,
    "name": "xgboost_credit_risk",
    "version": "v1_20250402_224915",
    "metrics": "{\"accuracy\": 0.934, \"precision\": 0.965, \"recall\": 0.724, \"f1\": 0.827, \"roc_auc\": 0.943, \"confusion_matrix\": [[5058, 37], [393, 1029]]}"
  }
}
```

## API Response Explanation

### Health Check (`GET /api/health`)

**In simple words:** "Yes, the system is online and working."

### Data Summary (`GET /api/data/summary`)

**In simple words:** "You have X total loans in the database. Y are in good standing (non-default) and Z have defaulted."

### Get Data (`GET /api/data`)

**In simple words:** "Here's a batch of loans with all their details. There are X total loans spread across Y pages, and you're looking at page Z with N records."

### ML Prediction (`POST /api/ml/predict`)

**In simple words:** "This loan has an X% chance of being repaid (non-default) and only a Y% chance of default. We're using our XGBoost model version from [date], which has a Z% accuracy rate."

## Categorical Data Mappings

When working with the API, use these mappings between numeric codes and human-readable values:

### Home Ownership Types

| Code | Display Value | Meaning                               |
| ---- | ------------- | ------------------------------------- |
| 0    | OWN           | Borrower owns their home outright     |
| 1    | MORTGAGE      | Borrower has a mortgage on their home |
| 2    | RENT          | Borrower rents their home             |
| 3    | OTHER         | Other living arrangement              |

### Loan Purpose/Intent

| Code | Display Value      | Meaning                              |
| ---- | ------------------ | ------------------------------------ |
| 0    | EDUCATION          | Educational expenses (tuition, etc.) |
| 1    | MEDICAL            | Medical or healthcare expenses       |
| 2    | VENTURE            | Business venture funding             |
| 3    | PERSONAL           | Personal expenses or needs           |
| 4    | HOME IMPROVEMENT   | Home repairs or renovations          |
| 5    | DEBT CONSOLIDATION | Combining multiple debts             |

### Loan Grade

| Code | Display Value | Credit Quality                |
| ---- | ------------- | ----------------------------- |
| 0    | A             | Excellent credit quality      |
| 1    | B             | Good credit quality           |
| 2    | C             | Fair credit quality           |
| 3    | D             | Below average credit quality  |
| 4    | E             | Poor credit quality           |
| 5    | F             | Very poor credit quality      |
| 6    | G             | Extremely poor credit quality |

### Default History

| Code | Display Value | Meaning                  |
| ---- | ------------- | ------------------------ |
| 0    | N             | No previous defaults     |
| 1    | Y             | Has defaulted previously |

### Loan Status (Prediction Outcome)

| Code | Display Value | Meaning                             |
| ---- | ------------- | ----------------------------------- |
| 0    | Non-Default   | Loan expected to be repaid normally |
| 1    | Default       | Loan at high risk of default        |

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

## Business Problem Addressed

The Credit Risk Analytics platform addresses several critical business problems for financial institutions:

1. **High Default Rates**: Accurately identifies high-risk applicants before loan approval, reducing financial losses.
2. **Inefficient Risk Assessment**: Automates credit risk evaluation for faster, more consistent decisions.
3. **Poor Resource Allocation**: Prioritizes loan applications with higher probability of repayment.
4. **Risk Pattern Identification**: Reveals combinations of factors that contribute most to defaults.
5. **Risk-Based Pricing**: Enables precise interest rate setting based on risk profiles.

## Dependencies

Main dependencies:

- Flask - Web framework
- SQLAlchemy - ORM
- Pandas - Data processing
- Scikit-learn - Machine learning
- XGBoost - Gradient boosting
- Flask-Swagger-UI - API documentation

See `requirements.txt` for the complete list.

## Data Source

This project uses the [Credit Risk Dataset](https://www.kaggle.com/laotse/credit-risk-dataset) from Kaggle. The raw CSV file should be placed in the `data/raw/` directory.

## Swagger Documentation

The API includes interactive documentation via Swagger UI:

1. Navigate to `/api/docs` in your browser
2. Explore all endpoints, parameters, and response formats
3. Test API calls directly from the UI

Swagger provides a comprehensive interface for understanding and experimenting with the API.
