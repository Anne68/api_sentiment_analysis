import streamlit as st
import requests
import pandas as pd
from io import StringIO

# URL de l'API FastAPI
api_url = "https://streamlitsentiment-e9dpc6graxehejhb.francecentral-01.azurewebsites.net"  # Modifiez si nécessaire

st.title("Analyse de Sentiment")

# Upload du fichier TSV
file = st.file_uploader("Téléchargez un fichier TSV", type=["tsv"])

if file is not None:
    # Affichage du fichier TSV téléchargé
    df = pd.read_csv(file, sep='\t')
    st.write("Aperçu du fichier original :")
    st.dataframe(df.head())

    # Bouton pour déclencher l'analyse des sentiments
    if st.button("Analyser les sentiments"):
        # Envoyer le fichier à l'API pour analyse
        response = requests.post(f"{api_url}/predict-sentiment/", files={"file": file.getvalue()})

        if response.status_code == 200:
            # Lire la réponse de l'API et afficher le fichier modifié
            updated_tsv = StringIO(response.text)
            df_updated = pd.read_csv(updated_tsv, sep='\t')
            st.write("Fichier nettoyé et prédictions ajoutées :")
            st.dataframe(df_updated.head())

            st.write(f"Taille du fichier après nettoyage et prédictions : {df_updated.shape}")

            # Télécharger le fichier modifié
            st.download_button(
                label="Télécharger le fichier modifié",
                data=response.text,
                file_name="fichier_avec_sentiment.tsv",
                mime="text/tsv"
            )
        else:
            st.error(f"Erreur lors de l'analyse des sentiments: {response.text}")
