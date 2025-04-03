import click
import os
import pandas as pd
import logging
from flask.cli import with_appcontext
from sqlalchemy import text

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def register_commands(app):
    """Register custom Flask CLI commands"""
    app.cli.add_command(init_db)
    app.cli.add_command(load_data)
    app.cli.add_command(etl_pipeline)
    app.cli.add_command(train_model)
    app.cli.add_command(data_summary)


@click.command("init-db")
@with_appcontext
def init_db():
    """Initialize the database"""
    from extensions import db  # Import from extensions instead of app

    try:
        click.echo("Initializing database...")
        db.create_all()
        click.echo("Database initialized successfully!")
    except Exception as e:
        click.echo(f"Error initializing database: {str(e)}")


@click.command("load-data")
@click.option("--file", "-f", default=None, help="Path to CSV file")
@click.option(
    "--sample", "-s", type=int, default=None, help="Number of records to sample"
)
@with_appcontext
def load_data(file, sample):
    """Load data from CSV into the database"""
    from extensions import db  # Import from extensions
    from models.credit_risk import CreditRisk
    from etl.extract import extract_data
    from etl.transform import transform_data
    from flask import current_app  # Use this to access app.config

    try:
        click.echo("Loading data...")

        # Extract data
        df = extract_data(file_path=file, sample_size=sample)

        # Transform data
        df = transform_data(df)

        # Clear existing data
        click.echo("Clearing existing data...")
        db.session.query(CreditRisk).delete()
        db.session.commit()

        # Load data into database
        click.echo(f"Loading {len(df)} records into database...")

        # Convert DataFrame to dictionary and bulk insert
        records = df.to_dict(orient="records")
        db.session.bulk_insert_mappings(CreditRisk, records)
        db.session.commit()

        click.echo("Data loaded successfully!")

    except Exception as e:
        db.session.rollback()
        click.echo(f"Error loading data: {str(e)}")


@click.command("etl-pipeline")
@click.option("--file", "-f", default=None, help="Path to CSV file")
@click.option(
    "--sample", "-s", type=int, default=None, help="Number of records to sample"
)
@click.option(
    "--output", "-o", default=None, help="Output file path for processed data"
)
@with_appcontext
def etl_pipeline(file, sample, output):
    """Run the full ETL pipeline"""
    from etl.extract import extract_data
    from etl.transform import transform_data
    from etl.load import load_to_db, save_to_csv
    from extensions import db  # Import from extensions
    from flask import current_app  # Use this to access app config

    try:
        click.echo("Starting ETL pipeline...")

        # Extract data
        click.echo("Extracting data...")
        df = extract_data(file_path=file, sample_size=sample)

        # Transform data
        click.echo("Transforming data...")
        df_transformed = transform_data(df)

        # Load data to database
        click.echo("Loading data to database...")
        load_to_db(df_transformed, db)

        # Save processed data to CSV if output path provided
        if output:
            click.echo(f"Saving processed data to {output}...")
            save_to_csv(df_transformed, output)
        else:
            # Use default output path from config
            default_output = os.path.join(
                current_app.config.get("PROCESSED_DATA_DIR", "data/processed"),
                "processed_credit_risk_data.csv",
            )
            click.echo(f"Saving processed data to {default_output}...")
            save_to_csv(df_transformed, default_output)

        click.echo("ETL pipeline completed successfully!")

    except Exception as e:
        click.echo(f"Error in ETL pipeline: {str(e)}")


@click.command("train-model")
@click.option("--input", "-i", default=None, help="Input file path for processed data")
@click.option(
    "--model-type",
    "-m",
    default="xgboost",
    help="Model type (xgboost, random_forest, logistic)",
)
@with_appcontext
def train_model(input, model_type):
    """Train a machine learning model"""
    from ml.train import train_model_pipeline
    from extensions import db  # Import from extensions
    from models.credit_risk import MLModel
    from flask import current_app  # Use this to access app config
    import json

    try:
        click.echo(f"Training {model_type} model...")

        # Use default input path if not provided
        if input is None:
            input = os.path.join(
                current_app.config.get("PROCESSED_DATA_DIR", "data/processed"),
                "processed_credit_risk_data.csv",
            )

        # Train model
        model_info = train_model_pipeline(
            input_file=input,
            model_type=model_type,
            model_dir=current_app.config.get("MODEL_DIR", "ml/models"),
        )

        # Save model info to database
        model = MLModel(
            name=model_info["name"],
            version=model_info["version"],
            model_type=model_info["model_type"],
            model_path=model_info["model_path"],
            metrics=json.dumps(model_info["metrics"]),
            parameters=json.dumps(model_info["parameters"]),
        )

        db.session.add(model)
        db.session.commit()

        click.echo("Model trained and saved successfully!")
        click.echo(f'Model metrics: {model_info["metrics"]}')

    except Exception as e:
        db.session.rollback()
        click.echo(f"Error training model: {str(e)}")


@click.command("data-summary")
@with_appcontext
def data_summary():
    """Print a summary of the data in the database"""
    from extensions import db  # Import from extensions
    from models.credit_risk import CreditRisk

    try:
        # Count total records
        count = db.session.query(CreditRisk).count()
        click.echo(f"Total records: {count}")

        # If no records, exit early
        if count == 0:
            click.echo("No records in database. Please load data first.")
            return

        # Count by loan status
        loan_status_counts = (
            db.session.query(CreditRisk.loan_status, db.func.count(CreditRisk.id))
            .group_by(CreditRisk.loan_status)
            .all()
        )

        click.echo("\nLoan Status Distribution:")
        for status, status_count in loan_status_counts:
            status_display = status if status is not None else "Unknown"
            click.echo(f"  {status_display}: {status_count} records")

        # Show some basic statistics
        click.echo("\nBasic Statistics:")
        person_age_stats = db.session.query(
            db.func.min(CreditRisk.person_age),
            db.func.max(CreditRisk.person_age),
            db.func.avg(CreditRisk.person_age),
        ).first()

        # Handle potentially None values
        min_age = person_age_stats[0] if person_age_stats[0] is not None else "N/A"
        max_age = person_age_stats[1] if person_age_stats[1] is not None else "N/A"
        avg_age = (
            f"{person_age_stats[2]:.2f}" if person_age_stats[2] is not None else "N/A"
        )

        click.echo(f"  Person Age: min={min_age}, max={max_age}, avg={avg_age}")

        loan_amnt_stats = db.session.query(
            db.func.min(CreditRisk.loan_amnt),
            db.func.max(CreditRisk.loan_amnt),
            db.func.avg(CreditRisk.loan_amnt),
        ).first()

        # Handle potentially None values
        min_loan = f"${loan_amnt_stats[0]}" if loan_amnt_stats[0] is not None else "N/A"
        max_loan = f"${loan_amnt_stats[1]}" if loan_amnt_stats[1] is not None else "N/A"
        avg_loan = (
            f"${loan_amnt_stats[2]:.2f}" if loan_amnt_stats[2] is not None else "N/A"
        )

        click.echo(f"  Loan Amount: min={min_loan}, max={max_loan}, avg={avg_loan}")

    except Exception as e:
        click.echo(f"Error getting data summary: {str(e)}")
