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
