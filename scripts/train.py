import mlflow
import mlflow.sklearn
import pandas as pd
import dvc.api
import yaml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Configurer MLflow
mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])

# Charger les données via DVC
data_url = dvc.api.get_url(params["data"]["path"], remote=params["data"]["remote"])
df = pd.read_parquet(data_url)
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params["model"]["test_size"], random_state=params["model"]["random_state"])

# Signature
input_example = X_train.head(1)

# --------------------------
# Grid Search LogisticRegression
# --------------------------
for C in params["model"]["logistic_regression"]["C"]:
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
for n in params["model"]["random_forest"]["n_estimators"]:
    with mlflow.start_run(run_name=f"RandomForest_n={n}"):
        model = RandomForestClassifier(n_estimators=n, random_state=params["model"]["random_state"])
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