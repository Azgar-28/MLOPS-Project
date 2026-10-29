import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_preprocessing import load_data, split_data
from drift_detection import detect_drift

mlflow.set_tracking_uri(
    "sqlite:///C:/Users/azgar/PycharmProjects/PythonProject1/mlflow.db"
)

mlflow.set_experiment("Automated_Model_Retraining")


def retrain_model():

    print("\n================================")
    print("        MODEL RETRAINING")
    print("================================")

    # Load original dataset
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model parameters
    n_estimators = 100
    random_state = 42

    # Start MLflow run
    with mlflow.start_run(run_name="Automatic_Retraining"):

        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("New Model Accuracy:", accuracy)

        # Log parameters
        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_param(
            "random_state",
            random_state
        )

        mlflow.log_param(
            "training_reason",
            "Data Drift Detected"
        )

        # Log metric
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # Find project folder
        project_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Save model
        model_path = os.path.join(
            project_path,
            "models",
            "model.pkl"
        )

        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="retrained_model"
        )

        print("New model saved successfully!")
        print("MLflow retraining run logged!")
        print("Model path:", model_path)


def main():

    print("================================")
    print(" AUTOMATED MODEL RETRAINING")
    print("================================")

    # Check for drift
    drift_detected = detect_drift()

    if drift_detected:

        print("\nWARNING: Data drift detected!")
        print("Starting automatic model retraining...")

        retrain_model()

    else:

        print("\nNo data drift detected.")
        print("Existing model will be used.")


if __name__ == "__main__":
    main()