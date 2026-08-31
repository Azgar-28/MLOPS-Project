import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_preprocessing import load_data, split_data

mlflow.set_tracking_uri(
    "sqlite:///C:/Users/azgar/PycharmProjects/PythonProject1/mlflow.db"
)

mlflow.set_experiment("Automated_Model_Retraining")


def train_model():

    print("================================")
    print("        MODEL TRAINING")
    print("================================")

    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model parameters
    n_estimators = 100
    random_state = 42

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Start MLflow run
    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("Model Accuracy:", accuracy)

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

        # Log metric
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # Save model locally
        project_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        model_path = os.path.join(
            project_path,
            "models",
            "model.pkl"
        )

        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model"
        )

        print("Model saved successfully!")
        print("MLflow tracking completed!")


if __name__ == "__main__":
    train_model()