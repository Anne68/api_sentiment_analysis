from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from preprocessing import nettoyage_automatisé  # On suppose que preprocessing gère déjà tout
from langdetect import detect, LangDetectException
import io

app = FastAPI()

# Charger le modèle
model = joblib.load("bernoulli_model.joblib")

def is_english(text: str) -> bool:
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False

# Via drag and drop
@app.post("/predict-csv/")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    # Lire le fichier CSV/TSV
    df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')

    # Appliquer le preprocessing (aucune nouvelle colonne n'est créée ici)
    df_cleaned = nettoyage_automatisé(df)

    # Faire la prédiction (le modèle sait déjà sur quelle colonne il doit travailler)
    predictions = model.predict(df_cleaned)

    # Retourner les résultats avec les prédictions
    return {"predictions": predictions.tolist()}

# Via phrase entrée à la mano
@app.post("/predict-text/")
async def predict_text(text: str):
    if len(text) < 50:
        return {"error": "Le commentaire doit contenir au moins 50 caractères"}

    if not is_english(text):
        return {"error": "Le commentaire n'est pas en anglais. Veuillez entrer un commentaire en anglais."}

    # Appliquer le preprocessing à la phrase donnée
    cleaned_text = nettoyage_automatisé(pd.DataFrame([text], columns=['Phrase']))

    # Faire la prédiction
    prediction = model.predict(cleaned_text)[0]

    return {"prediction": prediction}
