# backend/swagger.py
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from flask_swagger_ui import get_swaggerui_blueprint

# Create APISpec
spec = APISpec(
    title="Credit Risk Analytics API",
    version="1.0.0",
    openapi_version="3.0.2",
    plugins=[MarshmallowPlugin()],
    info=dict(
        description="API for Credit Risk Analytics and Prediction",
        contact=dict(email="your.email@example.com"),
    ),
)

# Define schemas for request/response models
spec.components.schema(
    "HealthResponse",
    {
        "type": "object",
        "properties": {"status": {"type": "string"}, "message": {"type": "string"}},
    },
)

spec.components.schema(
    "DataSummaryResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "object",
                "properties": {
                    "total_records": {"type": "integer"},
                    "loan_status_distribution": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
            },
        },
    },
)

spec.components.schema(
    "CreditRiskData",
    {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "person_age": {"type": "integer"},
            "person_income": {"type": "integer"},
            "person_home_ownership": {
                "type": "integer",
                "description": "0=OWN, 1=MORTGAGE, 2=RENT, 3=OTHER",
            },
            "person_emp_length": {"type": "number"},
            "loan_intent": {
                "type": "integer",
                "description": "0=EDUCATION, 1=MEDICAL, 2=VENTURE, 3=PERSONAL, 4=HOMEIMPROVEMENT, 5=DEBTCONSOLIDATION",
            },
            "loan_grade": {
                "type": "integer",
                "description": "0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G",
            },
            "loan_amnt": {"type": "integer"},
            "loan_int_rate": {"type": "number"},
            "loan_status": {
                "type": "integer",
                "description": "0=Non-Default, 1=Default",
            },
            "loan_percent_income": {"type": "number"},
            "cb_person_default_on_file": {"type": "integer", "description": "0=N, 1=Y"},
            "cb_person_cred_hist_length": {"type": "integer"},
            "debt_to_income": {"type": "number"},
            "income_to_loan_ratio": {"type": "number"},
        },
    },
)

spec.components.schema(
    "PaginatedResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CreditRiskData"},
            },
            "pagination": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "per_page": {"type": "integer"},
                    "total_pages": {"type": "integer"},
                    "total_records": {"type": "integer"},
                },
            },
        },
    },
)

spec.components.schema(
    "PredictionRequest",
    {
        "type": "object",
        "properties": {
            "person_age": {"type": "integer"},
            "person_income": {"type": "integer"},
            "person_home_ownership": {
                "type": "integer",
                "description": "0=OWN, 1=MORTGAGE, 2=RENT, 3=OTHER",
            },
            "person_emp_length": {"type": "number"},
            "loan_intent": {
                "type": "integer",
                "description": "0=EDUCATION, 1=MEDICAL, 2=VENTURE, 3=PERSONAL, 4=HOMEIMPROVEMENT, 5=DEBTCONSOLIDATION",
            },
            "loan_grade": {
                "type": "integer",
                "description": "0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G",
            },
            "loan_amnt": {"type": "integer"},
            "loan_int_rate": {"type": "number"},
            "loan_percent_income": {"type": "number"},
            "cb_person_default_on_file": {"type": "integer", "description": "0=N, 1=Y"},
            "cb_person_cred_hist_length": {"type": "integer"},
            "debt_to_income": {"type": "number"},
            "income_to_loan_ratio": {"type": "number"},
        },
        "required": [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
        ],
    },
)

spec.components.schema(
    "PredictionResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "prediction": {"type": "array", "items": {"type": "integer"}},
            "probability": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "status_interpretation": {"type": "array", "items": {"type": "string"}},
            "default_probability": {"type": "array", "items": {"type": "number"}},
            "model_info": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "metrics": {"type": "string"},
                },
            },
        },
    },
)

# Add paths (endpoints)
# Health Check
spec.path(
    path="/api/health",
    operations={
        "get": {
            "summary": "API Health Check",
            "description": "Verifies that the API is running and operational",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthResponse"}
                        }
                    },
                }
            },
            "tags": ["System"],
        }
    },
)

# Data Summary
spec.path(
    path="/api/data/summary",
    operations={
        "get": {
            "summary": "Get Data Summary",
            "description": "Returns summary statistics about the credit risk data",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/DataSummaryResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data"],
        }
    },
)

# Get Data
spec.path(
    path="/api/data",
    operations={
        "get": {
            "summary": "Get Paginated Data",
            "description": "Returns paginated credit risk data with optional filtering",
            "parameters": [
                {
                    "name": "page",
                    "in": "query",
                    "description": "Page number",
                    "schema": {"type": "integer", "default": 1},
                },
                {
                    "name": "per_page",
                    "in": "query",
                    "description": "Records per page",
                    "schema": {"type": "integer", "default": 100},
                },
                {
                    "name": "loan_status",
                    "in": "query",
                    "description": "Filter by loan status (0=non-default, 1=default)",
                    "schema": {"type": "integer"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PaginatedResponse"}
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data"],
        }
    },
)

# ML Prediction
spec.path(
    path="/api/ml/predict",
    operations={
        "post": {
            "summary": "Make Prediction",
            "description": "Makes credit risk predictions based on loan applicant data",
            "requestBody": {
                "description": "Loan applicant data",
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/PredictionRequest"}
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Successful prediction",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/PredictionResponse"
                            }
                        }
                    },
                },
                "400": {"description": "Bad request - missing or invalid input data"},
                "404": {"description": "No trained model available"},
                "500": {"description": "Internal server error"},
            },
            "tags": ["Machine Learning"],
        }
    },
)


# Schema for Loan Distribution response
spec.components.schema(
    "LoanDistributionItem",
    {
        "type": "object",
        "properties": {"range": {"type": "string"}, "count": {"type": "integer"}},
    },
)

spec.components.schema(
    "LoanDistributionResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/LoanDistributionItem"},
            },
        },
    },
)

# Schema for Default Rate by Month response
spec.components.schema(
    "DefaultRateMonthItem",
    {
        "type": "object",
        "properties": {"month": {"type": "string"}, "default_rate": {"type": "number"}},
    },
)

spec.components.schema(
    "DefaultRateMonthResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DefaultRateMonthItem"},
            },
        },
    },
)

# Schema for Default Rate by Grade response
spec.components.schema(
    "DefaultRateGradeItem",
    {
        "type": "object",
        "properties": {
            "grade": {"type": "string"},
            "default_rate": {"type": "number"},
            "total_loans": {"type": "integer"},
        },
    },
)

spec.components.schema(
    "DefaultRateGradeResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DefaultRateGradeItem"},
            },
        },
    },
)

# Schema for Correlation Matrix response
spec.components.schema(
    "CorrelationValue",
    {
        "type": "object",
        "properties": {
            "feature": {"type": "string"},
            "correlation": {"type": "number"},
        },
    },
)

spec.components.schema(
    "CorrelationFeature",
    {
        "type": "object",
        "properties": {
            "feature": {"type": "string"},
            "values": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CorrelationValue"},
            },
        },
    },
)

spec.components.schema(
    "CorrelationMatrixResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CorrelationFeature"},
            },
        },
    },
)

# Schema for Loan Counts response
spec.components.schema(
    "CategoryCount",
    {
        "type": "object",
        "properties": {"category": {"type": "string"}, "count": {"type": "integer"}},
    },
)

spec.components.schema(
    "LoanCountsResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "object",
                "properties": {
                    "home_ownership": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/CategoryCount"},
                    },
                    "loan_intent": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/CategoryCount"},
                    },
                },
            },
        },
    },
)

# Schema for Income Analysis response
spec.components.schema(
    "IncomeAnalysisItem",
    {
        "type": "object",
        "properties": {
            "income_range": {"type": "string"},
            "count": {"type": "integer"},
            "default_rate": {"type": "number"},
        },
    },
)

spec.components.schema(
    "IncomeAnalysisResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/IncomeAnalysisItem"},
            },
        },
    },
)

# Schema for Feature Importance response
spec.components.schema(
    "FeatureImportanceItem",
    {
        "type": "object",
        "properties": {"feature": {"type": "string"}, "importance": {"type": "number"}},
    },
)

spec.components.schema(
    "FeatureImportanceResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/FeatureImportanceItem"},
            },
            "model_info": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                },
            },
        },
    },
)

# Schema for ML Models response
spec.components.schema(
    "MLModelItem",
    {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "version": {"type": "string"},
            "model_type": {"type": "string"},
            "metrics": {"type": "string"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    },
)

spec.components.schema(
    "MLModelsResponse",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/MLModelItem"},
            },
        },
    },
)

# Now add the path definitions for the new endpoints

# Loan Distribution
spec.path(
    path="/api/data/loan-distribution",
    operations={
        "get": {
            "summary": "Get Loan Distribution",
            "description": "Returns distribution of loans by amount ranges",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/LoanDistributionResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data Visualization"],
        }
    },
)

# Default Rate by Month
spec.path(
    path="/api/data/default-rate-by-month",
    operations={
        "get": {
            "summary": "Get Default Rate by Month",
            "description": "Returns default rates for each month",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/DefaultRateMonthResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data Visualization"],
        }
    },
)

# Default Rate by Grade
spec.path(
    path="/api/data/default-rate-by-grade",
    operations={
        "get": {
            "summary": "Get Default Rate by Loan Grade",
            "description": "Returns default rates for each loan grade",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/DefaultRateGradeResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data Visualization"],
        }
    },
)

# Correlation Matrix
spec.path(
    path="/api/data/correlation-matrix",
    operations={
        "get": {
            "summary": "Get Correlation Matrix",
            "description": "Returns correlation matrix for key features",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/CorrelationMatrixResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data Visualization"],
        }
    },
)

# Loan Counts
spec.path(
    path="/api/data/loan-counts",
    operations={
        "get": {
            "summary": "Get Loan Counts by Category",
            "description": "Returns loan counts by various categories",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/LoanCountsResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data Visualization"],
        }
    },
)

# Income Analysis
spec.path(
    path="/api/data/income-analysis",
    operations={
        "get": {
            "summary": "Get Income Analysis",
            "description": "Returns analysis of loans by income ranges",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/IncomeAnalysisResponse"
                            }
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Data Visualization"],
        }
    },
)

# Feature Importance
spec.path(
    path="/api/ml/feature-importance",
    operations={
        "get": {
            "summary": "Get Feature Importance",
            "description": "Returns feature importance scores from the latest ML model",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/FeatureImportanceResponse"
                            }
                        }
                    },
                },
                "404": {"description": "No trained model available"},
                "500": {"description": "Internal server error"},
            },
            "tags": ["Machine Learning"],
        }
    },
)

# ML Models list
spec.path(
    path="/api/ml/models",
    operations={
        "get": {
            "summary": "Get ML Models",
            "description": "Returns list of available ML models",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/MLModelsResponse"}
                        }
                    },
                },
                "500": {"description": "Internal server error"},
            },
            "tags": ["Machine Learning"],
        }
    },
)

# Create Swagger UI Blueprint
SWAGGER_URL = "/api/docs"  # URL for exposing Swagger UI
API_URL = "/api/swagger.json"  # URL for serving OpenAPI spec

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "Credit Risk Analytics API",
        "validatorUrl": None,  # Disable online validator
    },
)
