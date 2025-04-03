import os
import sys
import logging
import json
from flask import (
    Flask,
    jsonify,
    request,
    render_template,
)
import requests
from dotenv import load_dotenv
from flask_cors import CORS

# Set base directory and load environment variables
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app(config_class=None):
    """Application factory function to create and configure the Flask app"""
    app = Flask(__name__)
    # Enable CORS for all routes
    CORS(app)
    app.url_map.strict_slashes = False
    # Configure the app
    if config_class is None:
        from config.config import DevelopmentConfig

        config_class = DevelopmentConfig

    app.config.from_object(config_class)

    # Configure database URI
    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST")
    DB_PORT = os.environ.get("DB_PORT")
    DB_NAME = os.environ.get("DB_NAME")

    # Configure PostgreSQL as the default, with SQLite as fallback
    if all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        SQLALCHEMY_DATABASE_URI = (
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    else:
        # Fallback to SQLite (for development)
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'dev.sqlite')}"

    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "testsupersecret@1")

    # Initialize extensions with the app
    from extensions import db

    db.init_app(app)

    # Register Swagger UI blueprint
    from swagger import swaggerui_blueprint, SWAGGER_URL

    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    # Register Flask CLI commands
    from commands import register_commands

    register_commands(app)

    # Register template filters
    @app.template_filter("tojson")
    def to_json(value, indent=None):
        return json.dumps(value, indent=indent)

    @app.template_filter("fromjson")
    def from_json(value):
        return json.loads(value)

    # Register routes
    register_routes(app)

    return app


def register_routes(app):
    """Register routes with the application"""
    # Import models after db initialization to avoid circular imports
    from models.credit_risk import CreditRisk, CreditRiskSchema, MLModel
    from swagger import spec
    from extensions import db

    @app.route("/api/swagger.json")
    def swagger_json():
        return jsonify(spec.to_dict())

    @app.route("/", methods=["GET"])
    def index():
        """Root page"""
        return render_template("index.html")

    @app.route("/predict", methods=["GET", "POST"])
    def predict_loan():
        """Loan default prediction page"""
        error = None
        result = None
        form_data = {}

        if request.method == "POST":
            try:
                # Get form data and convert to appropriate types
                form_data = {
                    "person_age": int(request.form.get("person_age")),
                    "person_income": int(request.form.get("person_income")),
                    "person_home_ownership": int(
                        request.form.get("person_home_ownership")
                    ),
                    "person_emp_length": float(request.form.get("person_emp_length")),
                    "loan_intent": int(request.form.get("loan_intent")),
                    "loan_grade": int(request.form.get("loan_grade")),
                    "loan_amnt": int(request.form.get("loan_amnt")),
                    "loan_int_rate": float(request.form.get("loan_int_rate")),
                    "cb_person_default_on_file": int(
                        request.form.get("cb_person_default_on_file")
                    ),
                    "cb_person_cred_hist_length": int(
                        request.form.get("cb_person_cred_hist_length")
                    ),
                    "debt_to_income": float(request.form.get("debt_to_income")),
                }

                # Calculate derived fields
                form_data["loan_percent_income"] = round(
                    form_data["loan_amnt"] / form_data["person_income"], 4
                )
                form_data["income_to_loan_ratio"] = round(
                    form_data["person_income"] / form_data["loan_amnt"], 2
                )

                # Make prediction request to our API
                api_url = request.url_root.rstrip("/") + "/api/ml/predict"
                response = requests.post(
                    api_url,
                    json=form_data,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get("status") == "success":
                        result = result_data
                    else:
                        error = result_data.get("message", "Prediction failed")
                else:
                    error = (
                        f"API request failed with status code: {response.status_code}"
                    )

            except ValueError as e:
                error = f"Invalid input: {str(e)}"
            except Exception as e:
                error = f"An error occurred: {str(e)}"

        # For GET request or if there was an error, just render the form
        return render_template(
            "prediction.html", error=error, result=result, form_data=form_data
        )

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """API Health Check endpoint"""
        return jsonify({"status": "success", "message": "API is running"})

    @app.route("/api/data", methods=["GET"])
    def get_data():
        """Get paginated data endpoint"""
        try:
            # Get query parameters
            page = request.args.get("page", 1, type=int)
            per_page = request.args.get("per_page", 100, type=int)
            loan_status = request.args.get("loan_status", None)

            # Build query
            query = CreditRisk.query

            # Apply filters if provided
            if loan_status:
                query = query.filter(CreditRisk.loan_status == loan_status)

            # Get paginated results
            paginated_data = query.paginate(page=page, per_page=per_page)

            # Serialize data
            schema = CreditRiskSchema(many=True)
            result = schema.dump(paginated_data.items)

            # Return response
            return jsonify(
                {
                    "status": "success",
                    "data": result,
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total_pages": paginated_data.pages,
                        "total_records": paginated_data.total,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error getting data: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/ml/predict", methods=["POST"])
    def predict():
        """Make prediction endpoint"""
        try:
            # Get input data from request
            data = request.json

            if not data:
                return (
                    jsonify({"status": "error", "message": "No input data provided"}),
                    400,
                )

            # Get the latest model
            model = MLModel.query.order_by(MLModel.created_at.desc()).first()

            if not model:
                return (
                    jsonify(
                        {"status": "error", "message": "No trained model available"}
                    ),
                    404,
                )

            # Preprocess input data and make prediction
            from ml.predict import preprocess_and_predict

            result = preprocess_and_predict(data, model.model_path)

            # Return prediction
            return jsonify(
                {
                    "status": "success",
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "model_info": {
                        "id": model.id,
                        "name": model.name,
                        "version": model.version,
                        "metrics": model.metrics,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # Admin Routes
    @app.route("/admin", methods=["GET"])
    def admin_dashboard():
        """Admin dashboard page"""
        try:
            # Get all records
            records = CreditRisk.query.order_by(CreditRisk.id.desc()).limit(1000).all()

            # Get count of all records
            total_records = CreditRisk.query.count()

            # Get count of default and non-default loans
            default_count = CreditRisk.query.filter_by(loan_status=1).count()
            non_default_count = CreditRisk.query.filter_by(loan_status=0).count()

            # Calculate default rate
            default_rate = 0
            if total_records > 0:
                default_rate = round((default_count / total_records) * 100, 2)

            return render_template(
                "admin.html",
                records=records,
                total_records=total_records,
                default_count=default_count,
                non_default_count=non_default_count,
                default_rate=default_rate,
            )
        except Exception as e:
            logger.error(f"Error loading admin dashboard: {str(e)}")
            return render_template("admin.html", error=str(e))

    @app.route("/admin/delete-all", methods=["POST"])
    def delete_all_records():
        """Delete all records from the database"""
        try:
            # Delete all records
            CreditRisk.query.delete()
            db.session.commit()

            return jsonify(
                {"status": "success", "message": "All records deleted successfully"}
            )
        except Exception as e:
            logger.error(f"Error deleting records: {str(e)}")
            db.session.rollback()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/data/summary", methods=["GET"])
    def get_data_summary():
        """Get data summary endpoint"""
        try:
            # Get count of records
            count = CreditRisk.query.count()

            # Get distribution of loan status
            loan_status_distribution = (
                db.session.query(CreditRisk.loan_status, db.func.count(CreditRisk.id))
                .group_by(CreditRisk.loan_status)
                .all()
            )

            # Format the distribution
            distribution = {
                str(status): count for status, count in loan_status_distribution
            }

            # Return the summary
            return jsonify(
                {
                    "status": "success",
                    "data": {
                        "total_records": count,
                        "loan_status_distribution": distribution,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500


# Create the app instance for running
app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
