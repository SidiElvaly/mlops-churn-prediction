
# 📊 MLOps Project: Telco Customer Churn Prediction

This project is a complete MLOps pipeline implementation to predict customer churn in a telecommunications company using machine learning. It integrates **data versioning**, **experiment tracking**, **automated training**, and **model registry** using tools such as **DVC**, **MLflow**, and more.

---

## 🎯 Objective

Build a scalable, reproducible, and collaborative machine learning pipeline that:
- Trains multiple classification models
- Tracks experiments and metrics
- Registers the best performing model
- Ensures reproducibility using DVC and Git
- Provides visualization and evaluation dashboards

---

## 📽️ Demo Video

You can watch the demo of this project here:  
👉 `mlops_projet.webm` (see repository or presentation platform)

---

## 🛠️ Tools & Technologies

| Tool        | Purpose                                      |
|-------------|----------------------------------------------|
| **Python**  | Programming language                         |
| **Pandas** / **Scikit-learn** | Data manipulation and ML models |
| **MLflow**  | Experiment tracking and model registry       |
| **DVC**     | Dataset version control                      |
| **Parquet** | Optimized dataset storage format             |
| **Git**     | Code versioning                              |
| **Matplotlib / Seaborn** | Evaluation and plotting        |

---

## 📂 Project Structure

```bash
.
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned & engineered dataset
├── models/               # Saved model files
├── scripts/
│   ├── preprocess.py     # Data cleaning and processing
│   ├── train.py          # Model training with MLflow logging
│   └── evaluate.py       # Evaluate best model and log artifacts
├── dvc.yaml              # DVC pipeline definition
├── mlruns/               # MLflow runs and metadata
└── README.md             # Project documentation
```

---

## 📊 Model Performance Summary

| Model              | Accuracy | F1 Score | AUC     |
|--------------------|----------|----------|---------|
| LogisticRegression | 0.79     | 0.72     | 0.83    |
| RandomForest_50    | 0.80     | 0.75     | 0.85    |
| RandomForest_100   | 0.81     | 0.76     | 0.86    |
| RandomForest_200   | **0.83** | **0.78** | **0.88** |

> ✅ The best model was **RandomForest_200** and it has been automatically registered to MLflow Model Registry.

---

## 🔁 Pipeline Steps

1. **Data Ingestion** using DVC with `data/raw/`
2. **Preprocessing** with `scripts/preprocess.py`
3. **Training & Logging** using MLflow via `scripts/train.py`
4. **Model Evaluation & Comparison** via `scripts/evaluate.py`
5. **Model Registry & Promotion** using MLflow Tracking Server

---

## 👥 Team Contributions

| Member         | Responsibilities                                     |
|----------------|------------------------------------------------------|
| Zidbih         | DVC setup, Data Preprocessing, pipeline automation   |
| Sidi El Valy   | Experiment tracking (MLflow), Model Evaluation       |
| Emani          | Model training & hyperparameter tuning               |
| Khatu          | Documentation, Report writing, Final presentation    |

---

## ✅ How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/<your-repo>/mlops-churn-prediction.git
cd mlops-churn-prediction
```

2. Create and activate virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Reproduce the pipeline:
```bash
dvc pull
dvc repro
```

4. Run training & evaluation:
```bash
python scripts/train.py
python scripts/evaluate.py
```

---

## 📦 Model Registry

Best model registered to:
```
Name: BestTelcoChurnModel
Alias: production
Tracked at: http://<MLflow-Tracking-Server>:5000
```

---

## 📌 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
