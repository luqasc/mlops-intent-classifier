import pytest
import sys
import os

# NÃO CARREGUE NADA NO TOPO DO ARQUIVO

@pytest.fixture(scope="function")
def integration_test_client(monkeypatch):
    """
    Fixture de integração que cria um ambiente de teste totalmente isolado.
    """
    # 1. Carrega o .env.test AQUI, DENTRO da fixture
    from dotenv import load_dotenv
    # O caminho é relativo à raiz do projeto, onde o pytest é executado
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env.test')
    load_dotenv(dotenv_path=dotenv_path, override=True)
    
    # 2. Usa monkeypatch para garantir que as variáveis lidas estão ativas
    monkeypatch.setenv("MONGO_URI", os.getenv("MONGO_URI"))
    monkeypatch.setenv("MONGO_DB", os.getenv("MONGO_DB"))
    monkeypatch.setenv("ENV", os.getenv("ENV"))

    # 3. Limpa o cache de módulos
    modules_to_clear = ["db.engine", "app.auth", "app.app"]
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

    # 4. Importa a aplicação com o ambiente limpo
    from fastapi.testclient import TestClient
    from app.app import app, collection

    # 5. Limpa o banco de dados de TESTE
    print(f"\n🧹 Limpando a collection '{collection.name}' no banco '{collection.database.name}'...")
    collection.delete_many({})

    with TestClient(app) as test_client:
        yield test_client, collection

    # --- TEARDOWN ---
    print(f"\n🧹 Limpando a collection '{collection.name}' após o teste...")
    collection.delete_many({})


def test_predict_endpoint_writes_to_db(integration_test_client):
    from unittest.mock import patch, MagicMock
    from bson import ObjectId
    
    client, collection = integration_test_client

    with patch("app.app.MODELS") as mock_models:
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ("certainty", {"certainty": 0.99})
        model_name = "integration-model"
        mock_models.items.return_value = [(model_name, mock_classifier)]

        input_text = "I am absolutely sure of this"
        response = client.post("/predict", params={"text": input_text})

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == input_text
    assert "id" in data

    record_id = ObjectId(data["id"])
    db_record = collection.find_one({"_id": record_id})

    assert db_record is not None
    assert db_record["text"] == input_text
    assert db_record["owner"] == "dev_user"