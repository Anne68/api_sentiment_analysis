from fastapi import FastAPI, UploadFile, File, Response
import pandas as pd
import joblib
from preprocessing import nettoyage_automatisé
import io

app = FastAPI()

# Charger le modèle
model = joblib.load("bernoulli_model.joblib")

@app.post("/predict-sentiment/")
async def predict_sentiment(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Lire le fichier TSV original
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')
        print(f"Taille du fichier original : {df.shape}")

        # Appliquer le prétraitement
        df_cleaned = nettoyage_automatisé(df)
        print(f"Taille après nettoyage : {df_cleaned.shape}")

        # Vérifier que la colonne textuelle 'cleaned_text' est présente
        if 'cleaned_text' not in df_cleaned.columns:
            return {"error": "Colonne textuelle 'cleaned_text' manquante après le nettoyage."}

        # Prédire le sentiment uniquement pour les lignes nettoyées
        predictions = model.predict(df_cleaned['cleaned_text'])
        print(f"Nombre de prédictions générées : {len(predictions)}")

        # Assurez-vous que les tailles correspondent
        if len(predictions) != len(df_cleaned):
            return {"error": f"Le nombre de prédictions ({len(predictions)}) ne correspond pas au nombre de lignes après nettoyage ({len(df_cleaned)})."}

        # Mapper les prédictions (0, 1 -> Négatif) et (3, 4 -> Positif)
        sentiment_labels = {0: 'Négatif', 1: 'Négatif', 3: 'Positif', 4: 'Positif'}
        df_cleaned['sentiment'] = [sentiment_labels.get(pred, 'Neutre') for pred in predictions]

        # Convertir le DataFrame nettoyé avec les prédictions en TSV
        output = df_cleaned.to_csv(sep='\t', index=False)

        # Renvoyer le fichier TSV nettoyé avec les prédictions
        return Response(content=output, media_type="text/tsv")
    except Exception as e:
        return {"error": str(e)}
