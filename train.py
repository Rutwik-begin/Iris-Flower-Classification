import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Create a folder for generated model files
Path("artifacts").mkdir(exist_ok=True)

# Load dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Flower_Classification")

# Models to compare
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_f1", macro_f1)

        report = classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        )
        mlflow.log_text(report, "classification_report.txt")

        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        trained_models[name] = model
        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

# Compare models using test metrics
results_df = pd.DataFrame(results).sort_values(
    ["accuracy", "macro_f1"],
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Save comparison table
results_df.to_csv("artifacts/model_comparison.csv", index=False)

# Select the best model according to the stated rule
best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "artifacts/iris_model.joblib")

metadata = {
    "model_name": best_name,
    "feature_names": list(X.columns),
    "target_names": list(iris.target_names)
}

with open("artifacts/model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

# Save the selected model's name in the console
print("\nSelected model:", best_name)
print("Saved model and metadata in the artifacts/ folder.")
print("MLflow tracking data is stored in ./mlruns/")
