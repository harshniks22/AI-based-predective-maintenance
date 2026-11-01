# AI-Based Predictive Maintenance System for Industrial Machines

## Project Overview

The AI-Based Predictive Maintenance System uses Machine Learning techniques to predict potential industrial machine failures based on sensor readings.

The system analyses machine operating parameters and predicts whether a machine failure is detected or not. A user-friendly web application is developed using Streamlit.

## Problem Statement

Unexpected industrial machine failures can cause production delays, increased maintenance costs, and reduced productivity.

Traditional maintenance methods may not identify potential failures early. Therefore, this project aims to develop an intelligent system that analyses machine sensor data and predicts possible failures using Machine Learning.

## Objectives

* Analyse industrial machine sensor data.
* Identify important machine operating parameters.
* Implement and compare Machine Learning algorithms.
* Predict potential machine failures.
* Develop an interactive Streamlit application.

## Dataset Description

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Dataset Source:** UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

* Total Records: 10,000
* Total Columns: 14
* Target Variable: Machine Failure
* Selected Input Features: 5

### Selected Features

1. Air Temperature (K)
2. Process Temperature (K)
3. Rotational Speed (RPM)
4. Torque (Nm)
5. Tool Wear (min)

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Plotly
* Joblib

## Machine Learning Algorithms

Three classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Performance

| Algorithm           |          Accuracy |
| ------------------- | ----------------: |
| Logistic Regression |            82.10% |
| Decision Tree       | Approximately 93% |
| Random Forest       |            98.15% |

Random Forest is selected for the final application based on its overall performance.

### Random Forest Evaluation

* Accuracy: 98.15%
* Precision: 73.13%
* Recall: 72.06%
* F1-Score: 72.59%

## Application Features

* Machine sensor input
* Machine failure prediction
* Model-estimated failure score
* Prediction history
* Interactive sensor reading graph
* CSV download option

## Project Structure

```text
AI-Based-Predictive-Maintenance/
│
├── app.py
├── main.py
├── random_forest_model.pkl
├── model_comparison.csv
├── requirements.txt
├── README.md
│
├── dataset/
│   └── ai4i2020.csv
│
└── screenshots/
```

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd AI-Based-Predictive-Maintenance
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## Application Screenshots

Screenshots of the developed application and model evaluation results will be included in this repository.

## Live Application

**Deployment Platform:** Streamlit Community Cloud

**Live Application URL:** To be added after deployment.

## Limitations

* The model uses a synthetic predictive maintenance dataset.
* Prediction results depend on the quality of input sensor values.
* The system predicts failure status but does not estimate the exact time until failure.

## Future Enhancements

* Real-time machine sensor monitoring.
* IoT device integration.
* Automated failure alerts.
* Improved predictions using real-world industrial datasets.

## Conclusion

This project demonstrates how Artificial Intelligence and Machine Learning can support predictive maintenance by identifying potential industrial machine failures using sensor data.

The Random Forest model is integrated into a Streamlit application to provide an interactive machine failure prediction system.
