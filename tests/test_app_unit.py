def test_health_endpoint():
    """Teste simples do endpoint /health"""
    from fastapi.testclient import TestClient
    from app.app import app
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_mocked():
    # Os patches devem ser aplicados ANTES de importar a aplicação
    from unittest.mock import patch, MagicMock
    
    # Patching collection e MODELS antes de qualquer coisa
    with patch("app.app.collection") as mock_collection, \
         patch("app.app.MODELS") as mock_models:
        
        # Agora, com os patches ativos, importamos a app e o client
        from fastapi.testclient import TestClient
        from app.app import app
        from bson import ObjectId
        
        client = TestClient(app)

        # Configuração dos mocks (sem alterações)
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = (
            "confusion", {"confusion": 0.92}
        )
        model_name_no_teste = "confusion-v1"
        mock_models.items.return_value = [(model_name_no_teste, mock_classifier)]

        fake_mongo_id = ObjectId()
        def insert_side_effect(document):
            document['_id'] = fake_mongo_id
            return None
        mock_collection.insert_one.side_effect = insert_side_effect

        # Execução
        response = client.post("/predict", params={"text": "what do you mean?"})

        # Verificações
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(fake_mongo_id)
        assert data["predictions"][model_name_no_teste]["top_intent"] == "confusion"
        mock_collection.insert_one.assert_called_once()