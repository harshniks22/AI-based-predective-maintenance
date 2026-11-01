import pandas as pd

# Load dataset
df = pd.read_csv("dataset/ai4i2020.csv")

# 1. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 2. Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# 3. Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# 4. Machine failure distribution
print("\nMACHINE FAILURE COUNT:")
print(df["Machine failure"].value_counts())

# 5. Machine failure percentage
print("\nMACHINE FAILURE PERCENTAGE:")
print(df["Machine failure"].value_counts(normalize=True) * 100)
import matplotlib.pyplot as plt
import seaborn as sns

# Machine failure count plot
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Machines")

plt.savefig("machine_failure.png", dpi=300)

plt.show()
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Air temperature [K]",
    bins=30,
    kde=True
)

plt.title("Air Temperature Distribution")
plt.xlabel("Air Temperature (Kelvin)")
plt.ylabel("Frequency")

plt.savefig("air_temperature_distribution.png", dpi=300)
plt.show()

# Graph 3: Process Temperature vs Air Temperature

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Air temperature [K]"],
    df["Process temperature [K]"],
    alpha=0.4,
    s=20,
    label="Machine Temperature Records"
)

plt.title("Relationship Between Air and Process Temperature")
plt.xlabel("Air Temperature (Kelvin)")
plt.ylabel("Process Temperature (Kelvin)")

plt.legend(title="Legend", loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("temperature_relationship.png", dpi=300)
plt.show()
# Graph 4: Rotational Speed vs Machine Failure

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Machine failure",
    y="Rotational speed [rpm]",
    hue="Machine failure",
    palette={0: "skyblue", 1: "salmon"},
    legend=False
)

plt.title("Rotational Speed vs Machine Failure")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Rotational Speed (RPM)")

plt.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color="skyblue", label="No Failure"),
        plt.Rectangle((0, 0), 1, 1, color="salmon", label="Failure")
    ],
    title="Machine Status"
)

plt.tight_layout()
plt.savefig("rotational_speed_failure.png", dpi=300)
plt.show()
# Graph 5: Average Torque vs Machine Failure

plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="Machine failure",
    y="Torque [Nm]",
    hue="Machine failure",
    palette={0: "skyblue", 1: "salmon"},
    errorbar=None,
    legend=False
)

plt.title("Average Torque vs Machine Failure")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Average Torque (Nm)")

plt.xticks([0, 1], ["No Failure", "Failure"])

plt.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color="skyblue", label="No Failure"),
        plt.Rectangle((0, 0), 1, 1, color="salmon", label="Failure")
    ],
    title="Machine Status"
)

plt.tight_layout()
plt.savefig("torque_failure.png", dpi=300)
plt.show()
# Graph 6: Average Tool Wear vs Machine Failure

tool_wear_avg = df.groupby("Machine failure")["Tool wear [min]"].mean()

plt.figure(figsize=(9, 6))

plt.plot(
    ["No Failure", "Failure"],
    tool_wear_avg.values,
    marker="o",
    linewidth=2,
    markersize=8,
    label="Average Tool Wear"
)

plt.title("Average Tool Wear vs Machine Failure")
plt.xlabel("Machine Status")
plt.ylabel("Average Tool Wear (Minutes)")

plt.legend(title="Legend")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("tool_wear_failure.png", dpi=300)
plt.show()
# MACHINE LEARNING - DATA PREPARATION

X = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

y = df["Machine failure"]

print("\nINPUT DATA SHAPE:", X.shape)
print("TARGET DATA SHAPE:", y.shape)
# STEP 2: SPLIT DATA INTO TRAINING AND TESTING

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)
# STEP 3: TRAIN LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression

# Create the model
log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
log_model.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION MODEL TRAINED SUCCESSFULLY!")
# STEP 4: MAKE PREDICTIONS

y_pred = log_model.predict(X_test)

print("\nFIRST 10 ACTUAL VALUES:")
print(y_test.head(10).values)

print("\nFIRST 10 PREDICTED VALUES:")
print(y_pred[:10])
# STEP 5: EVALUATE LOGISTIC REGRESSION MODEL

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("\nLOGISTIC REGRESSION MODEL PERFORMANCE")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, zero_division=0))
# STEP 6: TRAIN DECISION TREE MODEL

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Make predictions
dt_pred = dt_model.predict(X_test)

print("\nDECISION TREE MODEL TRAINED SUCCESSFULLY!")
# STEP 7: EVALUATE DECISION TREE MODEL

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("\nDECISION TREE MODEL PERFORMANCE")

print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, zero_division=0))
print("Recall:", recall_score(y_test, dt_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, dt_pred, zero_division=0))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, dt_pred, zero_division=0))
# STEP 8: TRAIN RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST MODEL TRAINED SUCCESSFULLY!")
# STEP 9: EVALUATE RANDOM FOREST MODEL

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("\nRANDOM FOREST MODEL PERFORMANCE")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, zero_division=0))
print("Recall:", recall_score(y_test, rf_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, rf_pred, zero_division=0))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, rf_pred, zero_division=0))
# STEP 10: MODEL COMPARISON

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, dt_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, dt_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, dt_pred),
        f1_score(y_test, rf_pred)
    ]
})

print("\nMODEL COMPARISON TABLE:")
print(comparison.round(3))

comparison.to_csv("model_comparison.csv", index=False)
# STEP 11: RANDOM FOREST CONFUSION MATRIX

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, rf_pred)

print("\nRANDOM FOREST CONFUSION MATRIX:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig("random_forest_confusion_matrix.png", dpi=300)
plt.show()
# STEP 12: SAVE THE BEST MACHINE LEARNING MODEL

import joblib

joblib.dump(rf_model, "random_forest_model.pkl")

print("\nRANDOM FOREST MODEL SAVED SUCCESSFULLY!")