import streamlit as st
import requests
import pandas as pd

# URL de l'API FastAPI
api_url = "http://127.0.0.1:8001"  # Assurez-vous que c'est l'URL correcte

# Configuration des styles Streamlit
st.set_page_config(page_title="Prédiction de Sentiment", layout="wide")

# Définir le style personnalisé
def set_bg_hack():
    st.markdown("""
        <style>
        .stApp {
            background-color: black;
            color: white;
        }
        .stButton>button {
            background-color: grey;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

# Appliquer le style personnalisé
set_bg_hack()

# Créer des onglets pour les différentes options (fichier ou texte)
tab1, tab2 = st.tabs(["Fichier CSV/TSV", "Commentaire texte"])

# Onglet pour l'upload de fichier
with tab1:
    st.header("Uploader un fichier CSV ou TSV")
    file = st.file_uploader("Choisir un fichier CSV ou TSV", type=["csv", "tsv"])

    if file is not None:
        # Lecture et affichage du fichier uploadé
        df = pd.read_csv(file, sep='\t')
        st.write("Aperçu du fichier :")
        st.dataframe(df.head())

        # Bouton pour déclencher la prédiction
        if st.button("Prédire à partir du fichier"):
            response = requests.post(f"{api_url}/predict-csv/", files={"file": file.getvalue()})
            if response.status_code == 200:
                predictions = response.json()
                st.write("Prédictions :")
                st.write(predictions)
            else:
                st.error("Erreur lors de la prédiction")

# Onglet pour entrer un commentaire texte
with tab2:
    st.header("Entrer un commentaire en anglais")

    # Champ texte pour le commentaire
    text = st.text_area("Entrez un commentaire (min. 50 caractères)", max_chars=500)

    if st.button("Prédire à partir du texte"):
        if len(text) < 50:
            st.warning("Le commentaire doit contenir au moins 50 caractères.")
        else:
            # Requête à l'API pour prédire le sentiment à partir du texte
            response = requests.post(f"{api_url}/predict-text/", json={"text": text})
            if response.status_code == 200:
                prediction = response.json()
                if "error" in prediction:
                    st.error(prediction["error"])
                else:
                    st.write(f"Prédiction : {prediction['prediction']}")
            else:
                st.error("Erreur lors de la prédiction")
