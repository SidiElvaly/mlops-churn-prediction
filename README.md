# Churn Prediction MLOps Pipeline

## Project Overview
This repository implements a complete MLOps pipeline to predict customer churn for a telecommunications service. It integrates:
- **Data Engineering** with DVC for reproducible data versioning and preprocessing.
- **Model Experimentation** using MLflow for tracking runs, parameters, metrics, and model registry.
- **Deployment** via a Flask web application, containerized and hosted on AWS EC2.

## Quickstart
```bash
# Clone the repository
git clone https://github.com/<org>/churn-prediction-mlops.git
cd churn-prediction-mlops
```

## Repository Structure
```
├── data-pipeline/         # Data ingestion, preprocessing, and DVC config
│   ├── data/              # Raw and processed datasets under DVC version control
│   ├── scripts/
│   │   └── preprocess.py  # Handles missing values, encoding, scaling
│   ├── dvc.yaml           # DVC pipeline definition
│   ├── params.yaml        # Preprocessing parameters
│   └── .dvcignore         # Patterns to ignore in DVC

├── model-training/        # Model development and MLflow tracking
│   ├── scripts/
│   │   ├── train.py       # Loads data, trains models (Logistic Regression, Random Forest)
│   │   └── evaluate.py    # Computes metrics (precision, F1, AUC) and saves plots
│   ├── mlruns/            # Local MLflow tracking directory
│   └── requirements.txt   # Dependencies for training and evaluation

├── web-app/               # Flask application for serving predictions
│   ├── webapp/
│   │   ├── app.py         # Flask server connecting to MLflow Model Registry
│   │   ├── templates/
│   │   │   └── form.html  # HTML form for input features
│   │   └── static/        # CSS and static assets
│   └── requirements.txt   # Flask and related dependencies

├── documentation/         # Architecture diagram and final report
│   ├── architecture.png   # Technical architecture overview
│   ├── rapport.pdf        # Final project report (PDF)
│   └── README.md          # Branch-specific documentation

├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Data Pipeline (Branch: `data-pipeline`)
Contains everything to reproduce and version the dataset.

**Usage:**
```bash
git checkout data-pipeline
pip install -r data-pipeline/requirements.txt
dvc pull
python data-pipeline/scripts/preprocess.py
```

**Details:**
- DVC tracks raw and cleaned data in `data/`.
- `preprocess.py` applies imputation, categorical encoding, and scaling.
- Parameters are centrally managed in `params.yaml`.

## Model Training (Branch: `model-training`)
Implements model fitting, evaluation, and MLflow integration.

**Usage:**
```bash
git checkout model-training
pip install -r model-training/requirements.txt
python model-training/scripts/train.py
python model-training/scripts/evaluate.py
```

**Details:**
- `train.py` logs hyperparameters and metrics to MLflow.
- Best model is automatically registered in the MLflow Model Registry.
- `evaluate.py` generates ROC curves and confusion matrices as artifacts.

## Web Application (Branch: `web-app`)
Provides a simple UI for real-time churn prediction.

**Usage:**
```bash
git checkout web-app
pip install -r web-app/requirements.txt
python webapp/app.py
```

**Details:**
- Connects to MLflow Tracking Server to load the latest production model.
- HTML form captures customer features and displays prediction.

## Documentation (Branch: `documentation`)
Hosts supplementary materials:
- `architecture.png`: Visual diagram of Git, DVC, S3, MLflow EC2, Flask EC2.
- `rapport.pdf`: Detailed report covering dataset, architecture, results, and lessons learned.

## Deployment Steps
1. **Configure AWS Credentials**
   ```bash
   aws configure
   ```
2. **Set up DVC Remote**
   ```bash
   dvc remote add -d s3remote s3://<your-bucket>/churn-pipeline
   dvc push
   ```
3. **Deploy MLflow Server** on an EC2 instance (port 5000). Use Gunicorn or the built-in server.
4. **Deploy Flask App** on a separate EC2 instance (port 80). Secure with a reverse proxy (e.g., Nginx).
5. Update environment variables (`MLFLOW_TRACKING_URI`, S3 credentials) on both servers.

## Branch Merging Guidelines

Follow these steps to merge branch changes into the main production branch:

1. **Switch to `main`**
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Merge the feature branch**
   ```bash
   git merge <branch-name>
   ```
3. **Resolve conflicts** (if any):
   - Open conflicted files marked by Git.
   - Edit to combine changes, then:
     ```bash
     git add <file-path>
     ```
4. **Commit the merge**
   ```bash
   git commit
   ```
5. **Push to remote**
   ```bash
   git push origin main
   ```

> _Tip_: Use pull requests (PRs) and code reviews before merging to ensure quality and track changes.

## Used Technologies

List of core technologies and tools:
- **Python**: Data processing, model training, and web application.
- **DVC**: Data version control for reproducible pipelines.
- **MLflow**: Experiment tracking, model registry, and artifact management.
- **Flask**: Lightweight web framework for serving predictions.
- **AWS S3**: Remote storage for DVC data and model artifacts.
- **EC2**: Hosting MLflow server and Flask application.
- **Git**: Version control system.
- **PostgreSQL**: Relational database for storing customer data and model metadata.
- **Pandas**, **NumPy**, **scikit-learn**, **Matplotlib**: Data manipulation, modeling, and visualization.
- **Gunicorn**: Application server for production deployment.