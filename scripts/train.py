import mlflow
import mlflow.sklearn
import pandas as pd
import dvc.api

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from mlflow.models.signature import infer_signature

# Config MLflow
mlflow.set_tracking_uri("http://3.222.185.201:5000")
mlflow.set_experiment("Projet MLOps - Churn Prediction")

# Chargement des données via DVC
data_url = dvc.api.get_url("data/processed/full.parquet", remote="s3remote")
df = pd.read_parquet(data_url)
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Signature
input_example = X_train.head(1)

# --------------------------
# Grid Search LogisticRegression
# --------------------------
for C in [0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"LogisticRegression_C={C}"):
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

# --------------------------
# Grid Search RandomForestClassifier
# --------------------------
for n in [50, 100, 200]:
    with mlflow.start_run(run_name=f"RandomForest_n={n}"):
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

