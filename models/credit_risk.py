from extensions import db  # Change this line
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from marshmallow import Schema, fields
from datetime import datetime


class CreditRisk(db.Model):
    """Credit Risk Data Model"""

    __tablename__ = "credit_risk"

    id = Column(Integer, primary_key=True)

    # Personal information
    person_age = Column(Integer)
    person_income = Column(Integer)
    person_home_ownership = Column(String(50))
    person_emp_length = Column(Float)

    # Loan information
    loan_intent = Column(String(50))
    loan_grade = Column(String(10))
    loan_amnt = Column(Integer)
    loan_int_rate = Column(Float)
    loan_status = Column(Integer)  # 0 = non-default, 1 = default
    loan_percent_income = Column(Float)

    # Credit history
    cb_person_default_on_file = Column(String(5))
    cb_person_cred_hist_length = Column(Integer)

    # Derived features (added during transformation)
    debt_to_income = Column(Float, nullable=True)
    income_to_loan_ratio = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CreditRisk {self.id}: ${self.loan_amnt} - Status: {self.loan_status}>"


class CreditRiskSchema(Schema):
    """Schema for serializing CreditRisk model"""

    id = fields.Integer(dump_only=True)
    person_age = fields.Integer()
    person_income = fields.Integer()
    person_home_ownership = fields.String()
    person_emp_length = fields.Float()
    loan_intent = fields.String()
    loan_grade = fields.String()
    loan_amnt = fields.Integer()
    loan_int_rate = fields.Float()
    loan_status = fields.Integer()
    loan_percent_income = fields.Float()
    cb_person_default_on_file = fields.String()
    cb_person_cred_hist_length = fields.Integer()
    debt_to_income = fields.Float()
    income_to_loan_ratio = fields.Float()
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)


class MLModel(db.Model):
    """Machine Learning Model Metadata"""

    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    version = Column(String(20))
    model_type = Column(String(50))  # xgboost, random_forest, etc.
    model_path = Column(String(255))  # path to saved model file
    metrics = Column(Text)  # JSON string of model metrics
    parameters = Column(Text)  # JSON string of model parameters
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MLModel {self.name} v{self.version}>"


class MLModelSchema(Schema):
    """Schema for serializing MLModel model"""

    id = fields.Integer(dump_only=True)
    name = fields.String()
    version = fields.String()
    model_type = fields.String()
    model_path = fields.String()
    metrics = fields.String()
    parameters = fields.String()
    created_at = fields.DateTime(dump_only=True)
