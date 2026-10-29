# Automated Model Retraining System Using Data Drift Detection

## 1. Project Overview

This project implements an automated machine learning monitoring and retraining workflow. It detects data drift in incoming data and triggers model retraining when drift is detected.

The system uses Python, Evidently, Scikit-learn, MLflow, Streamlit, and Watchdog to monitor data, manage model training, track experiments, and display system status.

## 2. Objectives

- Detect data drift between reference and incoming datasets.
- Generate an HTML drift detection report.
- Automatically trigger model retraining when drift is detected.
- Track training metrics and parameters using MLflow.
- Monitor incoming CSV files automatically.
- Display model and monitoring information through a Streamlit dashboard.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Evidently
- MLflow
- Streamlit
- Watchdog
- DVC
- Git and GitHub

## 4. Project Structure

```text
PythonProject1/
├── data/
│   ├── training_data.csv
│   ├── new_data.csv
│   └── drift_report.html
├── incoming_data/
├── models/
│   └── model.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── drift_detection.py
│   ├── retrain.py
│   └── train_model.py
├── dashboard.py
├── watcher.py
├── mlflow.db
├── requirements.txt
└── README.md
```

Note: The structure above describes the main project files. Some files, such as `requirements.txt`, should be created if they do not already exist.

## 5. System Workflow

1. The watcher monitors the `incoming_data` directory.
2. When a new CSV file arrives, the watcher waits for the file to finish copying.
3. The incoming CSV is copied to `data/new_data.csv`.
4. The drift detection module compares incoming data with the reference data.
5. If drift is detected, the retraining script starts.
6. A Random Forest classifier is trained and evaluated.
7. The trained model is saved as `models/model.pkl`.
8. MLflow records model parameters, accuracy, and run information.
9. The HTML drift report and monitoring status are made available to the dashboard.

## 6. Installation and Setup

### Step 1: Open the project directory

```powershell
cd C:\Users\azgar\PycharmProjects\PythonProject1
```

### Step 2: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

If `requirements.txt` is available:

```powershell
pip install -r requirements.txt
```

Otherwise, install the required packages:

```powershell
pip install pandas numpy scikit-learn joblib evidently mlflow streamlit watchdog dvc
```

## 7. Running the Project

Run each component in a separate terminal.

### Terminal 1: Start MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///C:/Users/azgar/PycharmProjects/PythonProject1/mlflow.db --port 5000
```

Open: http://localhost:5000

### Terminal 2: Start the Streamlit dashboard

```powershell
streamlit run dashboard.py
```

Open: http://localhost:8501

### Terminal 3: Start automatic file monitoring

```powershell
python watcher.py
```

Copy a CSV file into the `incoming_data` directory to trigger the workflow.

## 8. Experimental Results

The following results were observed during testing:

| Metric | Result |
|---|---|
| Reference dataset | 569 rows, 31 columns |
| Incoming dataset | 200 rows, 30 columns |
| Drifted columns | 2 |
| Drift share | 6.67% |
| Model | RandomForestClassifier |
| Number of estimators | 100 |
| Random state | 42 |
| Model accuracy | 96.49% |
| Model artifact | `models/model.pkl` |

The reported accuracy was calculated on a held-out split of the original labelled dataset. It does not represent evaluation on the incoming data.

## 9. MLflow Experiment Tracking

MLflow records:

- Model parameters
- Accuracy metric
- Training run information
- Logged model information and artifacts

Experiment name: `Automated_Model_Retraining`

Run name: `Automatic_Retraining`

## 10. Limitations

- The current retraining script trains on the original labelled dataset. Incoming CSV files are used for drift detection, not directly for training.
- Incoming data must be validated before it is used for future retraining.
- The watcher runs locally and must remain active to detect incoming files.
- The processed-file tracking currently lasts only for the active watcher session.
- Production deployment, external data ingestion, and persistent monitoring are future improvements.

## 11. Future Enhancements

- Incorporate validated, labelled incoming data into retraining.
- Add persistent duplicate-event tracking.
- Implement model performance monitoring and approval workflows.
- Add automatic dashboard refresh.
- Configure a DVC remote for dataset storage.
- Deploy the application to a cloud environment.
- Add automated testing and a CI/CD pipeline.

## 12. Conclusion

The project demonstrates a local automated machine learning workflow that detects data drift, triggers model retraining, logs experiments in MLflow, and provides a monitoring dashboard. It establishes a foundation for more robust and production-ready MLOps systems.
