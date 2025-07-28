import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api
import yaml

# Charger les paramètres depuis params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Configurer MLflow
mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])

# Charger les données versionnées par DVC
data_path = dvc.api.get_url(params["data"]["path"], remote=params["data"]["remote"])
df = pd.read_parquet(data_path)
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Récupérer les runs
runs = mlflow.search_runs(experiment_names=[params["mlflow"]["experiment_name"]])
best_f1 = 0
best_run_id = None

for _, run in runs.iterrows():
    with mlflow.start_run(run_id=run.run_id, nested=True):
        # Charger le modèle
        model = mlflow.sklearn.load_model(f"runs:/{run.run_id}/model")
        
        # Prédictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Calculer les métriques
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        # Logger les métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        
        # Générer et enregistrer la courbe ROC
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()
        
        # Générer et enregistrer la matrice de confusion
        cm = confusion_matrix(y_test, predictions)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Identifier le meilleur modèle
        if f1 > best_f1:
            best_f1 = f1
            best_run_id = run.run_id

# Enregistrer le meilleur modèle dans le Model Registry
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "BestTelcoChurnModel")
    print(f"Meilleur modèle enregistré avec F1-score {best_f1}")