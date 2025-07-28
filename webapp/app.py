# ==============================================================================
# FICHIER : webapp/app.py
# VERSION : SANS PROBABILITÉ (PRÉDICTION SIMPLE)
# ==============================================================================
from flask import Flask, render_template, request, jsonify
import pandas as pd
import mlflow
import os
import traceback

app = Flask(__name__)

# --- CONFIGURATION ET CHARGEMENT DU MODÈLE ---
MLFLOW_TRACKING_URI = "http://3.222.185.201:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "BestTelcoChurnModel"
MODEL_ALIAS = "production"
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = None

try:
    os.environ['AWS_NO_SIGN_REQUEST'] = 'true'
    print(f"Chargement du modèle : {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅  SUCCÈS : Le modèle a été chargé correctement.")
except Exception as e:
    print(f"🔴 ERREUR CRITIQUE : Impossible de charger le modèle. Erreur : {e}")

# --- FONCTION DE PRÉ-TRAITEMENT MANUEL (Inchangée) ---
def preprocess_input(data: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([data])
    for col in ['tenure', 'SeniorCitizen']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(int)
    for col in ['MonthlyCharges', 'TotalCharges']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
    expected_schema_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
        'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
        'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
        'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    processed_df = pd.DataFrame(0.0, index=[0], columns=expected_schema_cols)
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in input_df: processed_df[col] = input_df[col]
    if input_df['gender'].iloc[0] == 'Male': processed_df['gender_Male'] = 1.0
    if input_df['Partner'].iloc[0] == 'Yes': processed_df['Partner_Yes'] = 1.0
    if input_df['Dependents'].iloc[0] == 'Yes': processed_df['Dependents_Yes'] = 1.0
    if input_df['PhoneService'].iloc[0] == 'Yes': processed_df['PhoneService_Yes'] = 1.0
    if input_df['PaperlessBilling'].iloc[0] == 'Yes': processed_df['PaperlessBilling_Yes'] = 1.0
    if input_df['MultipleLines'].iloc[0] == 'No phone service': processed_df['MultipleLines_No phone service'] = 1.0
    elif input_df['MultipleLines'].iloc[0] == 'Yes': processed_df['MultipleLines_Yes'] = 1.0
    if input_df['InternetService'].iloc[0] == 'Fiber optic': processed_df['InternetService_Fiber optic'] = 1.0
    elif input_df['InternetService'].iloc[0] == 'No': processed_df['InternetService_No'] = 1.0
    if input_df['OnlineSecurity'].iloc[0] == 'No internet service': processed_df['OnlineSecurity_No internet service'] = 1.0
    elif input_df['OnlineSecurity'].iloc[0] == 'Yes': processed_df['OnlineSecurity_Yes'] = 1.0
    if input_df['OnlineBackup'].iloc[0] == 'No internet service': processed_df['OnlineBackup_No internet service'] = 1.0
    elif input_df['OnlineBackup'].iloc[0] == 'Yes': processed_df['OnlineBackup_Yes'] = 1.0
    if input_df['DeviceProtection'].iloc[0] == 'No internet service': processed_df['DeviceProtection_No internet service'] = 1.0
    elif input_df['DeviceProtection'].iloc[0] == 'Yes': processed_df['DeviceProtection_Yes'] = 1.0
    if input_df['TechSupport'].iloc[0] == 'No internet service': processed_df['TechSupport_No internet service'] = 1.0
    elif input_df['TechSupport'].iloc[0] == 'Yes': processed_df['TechSupport_Yes'] = 1.0
    if input_df['StreamingTV'].iloc[0] == 'No internet service': processed_df['StreamingTV_No internet service'] = 1.0
    elif input_df['StreamingTV'].iloc[0] == 'Yes': processed_df['StreamingTV_Yes'] = 1.0
    if input_df['StreamingMovies'].iloc[0] == 'No internet service': processed_df['StreamingMovies_No internet service'] = 1.0
    elif input_df['StreamingMovies'].iloc[0] == 'Yes': processed_df['StreamingMovies_Yes'] = 1.0
    if input_df['Contract'].iloc[0] == 'One year': processed_df['Contract_One year'] = 1.0
    elif input_df['Contract'].iloc[0] == 'Two year': processed_df['Contract_Two year'] = 1.0
    if input_df['PaymentMethod'].iloc[0] == 'Credit card (automatic)': processed_df['PaymentMethod_Credit card (automatic)'] = 1.0
    elif input_df['PaymentMethod'].iloc[0] == 'Electronic check': processed_df['PaymentMethod_Electronic check'] = 1.0
    elif input_df['PaymentMethod'].iloc[0] == 'Mailed check': processed_df['PaymentMethod_Mailed check'] = 1.0
    return processed_df

# --- ROUTES DE L'APPLICATION WEB (Inchangées) ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze')
def analyze_page():
    return render_template('form.html')

# --- ROUTE DE PRÉDICTION (MODIFIÉE) ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Le modèle n\'est pas disponible.'}), 500
    try:
        raw_data = request.get_json()
        processed_df = preprocess_input(raw_data)
        
        # <-- RETOUR À LA PRÉDICTION SIMPLE (0 ou 1) -->
        prediction_result = model.predict(processed_df)
        prediction = int(prediction_result[0])
        
        print(f"Résultat de la prédiction : {prediction}")

        # <-- On ne renvoie plus la confiance -->
        return jsonify({'prediction': prediction})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Erreur interne du serveur: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)