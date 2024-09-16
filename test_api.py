import pytest
from fastapi.testclient import TestClient
from api import app  # Importez votre application FastAPI

client = TestClient(app)

def test_prediction_route():
    # Simuler une requête POST avec un commentaire en entrée
    response = client.post("//home/vicky/nlp_sentiment/bernoulli_model.joblib", json={"comment": "This is a test comment with more than 50 characters."})

    # Vérifier que la réponse est 200 OK
    assert response.status_code == 200

    # Vérifier que le format de la réponse est correct
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], int)
    assert 0 <= response_json["prediction"] <= 4  # Par exemple, pour les classes de sentiment entre 0 et 4
