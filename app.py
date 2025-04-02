from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
import logging
import sys

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import DevelopmentConfig
from extensions import db  # Import the db instance from extensions

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
app.config.from_object(DevelopmentConfig)

# Configure SQLAlchemy
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///dev.sqlite"
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(BASE_DIR, 'dev.sqlite')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "testsupersecret@1"

# Initialize the db with the app
db.init_app(app)

# Import models and routes after db initialization to avoid circular imports
from models.credit_risk import CreditRisk, CreditRiskSchema, MLModel


@app.route("/", methods=["GET"])
def index():
    """Root page"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API is working"""
    return jsonify({"status": "success", "message": "API is running"})


@app.route("/api/data", methods=["GET"])
def get_data():
    """Get paginated data with optional filtering"""
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
    """Make predictions using the trained ML model"""
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
                jsonify({"status": "error", "message": "No trained model available"}),
                404,
            )

        # Preprocess input data and make prediction
        from ml.predict import preprocess_and_predict

        #        name = Column(String(100))
        # version = Column(String(20))
        # model_type = Column(String(50))  # xgboost, random_forest, etc.
        # model_path = Column(String(255))

        print(model.name)
        print(model.version)
        print(model.model_path)

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


# Data API Endpoints


@app.route("/api/data/summary", methods=["GET"])
def get_data_summary():
    """Return a summary of the credit risk data"""
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
        distribution = {status: count for status, count in loan_status_distribution}

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


# Register Flask CLI commands
from commands import register_commands

register_commands(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
