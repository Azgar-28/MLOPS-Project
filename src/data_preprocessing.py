from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def load_data():
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def create_data_files():

    X, y = load_data()

    # Find the main project folder
    project_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Create data folder path
    data_path = os.path.join(project_path, "data")

    # Create data folder if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Create training/reference data
    training_data = X.copy()
    training_data["target"] = y

    training_data.to_csv(
        os.path.join(data_path, "training_data.csv"),
        index=False
    )

    # Create new incoming data
    new_data = X.sample(
        n=200,
        random_state=10
    ).copy()
    # Simulate data drift
    new_data["mean radius"] = new_data["mean radius"] * 1.5
    new_data["mean texture"] = new_data["mean texture"] * 1.5

    new_data.to_csv(
        os.path.join(data_path, "new_data.csv"),
        index=False
    )

    print("Training data created!")
    print("New data created!")


if __name__ == "__main__":
    create_data_files()