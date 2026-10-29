
import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


def detect_drift():
    # Find project folder
    project_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    training_path = os.path.join(
        project_path, "data", "training_data.csv"
    )

    new_data_path = os.path.join(
        project_path, "data", "new_data.csv"
    )

    # Load datasets
    reference_data = pd.read_csv(training_path)
    current_data = pd.read_csv(new_data_path)

    print("Reference data shape:", reference_data.shape)
    print("New data shape:", current_data.shape)

    # Remove target from reference data
    if "target" not in reference_data.columns:
        raise ValueError(
            "Reference data is missing the 'target' column."
        )

    reference_features = reference_data.drop(
        columns=["target"]
    )

    # Ensure feature columns match
    if set(reference_features.columns) != set(current_data.columns):
        raise ValueError(
            "Reference and incoming feature columns do not match."
        )

    current_data = current_data[reference_features.columns]

    # Generate the Evidently drift report
    # Set threshold to 1% of features
    report = Report([
        DataDriftPreset(drift_share=0.01)
    ])

    # Current data first, reference data second
    result = report.run(
        current_data,
        reference_features
    )

    # Extract drift information
    result_dict = result.dict()
    drift_metric = result_dict["metrics"][0]
    drift_value = drift_metric["value"]

    drifted_columns = int(drift_value["count"])
    drift_share = float(drift_value["share"])

    # Drift is detected if at least one feature drifts
    drift_detected = drifted_columns > 0

    print("\n================================")
    print("       DRIFT DETECTION")
    print("================================")
    print("Drifted columns:", drifted_columns)
    print("Drift share:", drift_share)
    print("Rule: drift detected if at least one feature is drifted.")
    print(
        "Drift detected:",
        "YES" if drift_detected else "NO"
    )

    # Save the HTML report
    report_path = os.path.join(
        project_path, "data", "drift_report.html"
    )

    result.save_html(report_path)

    print("\nDrift report created successfully!")
    print("Report saved at:", report_path)

    return drift_detected


if __name__ == "__main__":
    detect_drift()
