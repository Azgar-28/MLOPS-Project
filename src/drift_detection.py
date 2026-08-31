import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


def detect_drift():

    # Find project folder
    project_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Dataset paths
    training_path = os.path.join(
        project_path,
        "data",
        "training_data.csv"
    )

    new_data_path = os.path.join(
        project_path,
        "data",
        "new_data.csv"
    )

    # Load datasets
    reference_data = pd.read_csv(training_path)
    current_data = pd.read_csv(new_data_path)

    print("Reference data shape:", reference_data.shape)
    print("New data shape:", current_data.shape)

    # Remove target from reference data
    reference_features = reference_data.drop(
        columns=["target"]
    )

    # Create drift report
    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        current_data,
        reference_features
    )

    # Get result as dictionary
    result_dict = result.dict()

    # Find number of drifted columns
    drift_metric = result_dict["metrics"][0]

    drifted_columns = int(
        drift_metric["value"]["count"]
    )

    drift_share = drift_metric["value"]["share"]

    # Dataset drift decision
    drift_detected = drifted_columns > 0

    print("\n================================")
    print("       DRIFT DETECTION")
    print("================================")

    print("Drifted columns:", drifted_columns)
    print("Drift share:", drift_share)

    if drift_detected:
        print("Drift detected: YES")
    else:
        print("Drift detected: NO")

    # Save HTML report
    report_path = os.path.join(
        project_path,
        "data",
        "drift_report.html"
    )

    result.save_html(report_path)

    print("\nDrift report created successfully!")
    print("Report saved at:", report_path)

    return drift_detected


if __name__ == "__main__":
    detect_drift()