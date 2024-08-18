# tests/test_main.py
from fastapi.testclient import TestClient
from application import app
import logging, pytest, random, pickle, neat, base64
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator

client = TestClient(app)

def test_receive_user_data():
    response = client.post("/user_data", json={"genome_id": 1, "time_since_startup": 123.45, "user_rating": 4})
    assert response.status_code == 200
    assert response.json() == {"message": "Data received successfully"}

def test_receive_user_data_invalid():
    response = client.post("/restart_population")
    response = client.post("/user_data", json={"genome_id": "invalid", "time_since_startup": "invalid", "user_rating": "invalid"})
    assert response.status_code == 422  # Unprocessable Entity

def test_receive_user_data_missing_fields():
    response = client.post("/restart_population")
    response = client.post("/user_data", json={"genome_id": 1})
    assert response.status_code == 422  # Unprocessable Entity

def test_request_new_mediator():
    response = client.post("/restart_population")
    response = client.post("/request_new_mediator", json={"genome_id": 1, "time_since_startup": 123.45, "user_rating": 4})
    assert response.status_code == 200, "Response status code is not 200"

    response_json = response.json()
    assert "new_mediator" in response_json, "Response does not contain new_mediator"
    new_mediator = response_json["new_mediator"]
    try:
        pickled_object = base64.b64decode(new_mediator)
    except (TypeError, ValueError):
        assert False, "new_mediator is not a valid base64-encoded string"
    try:
        genome_id, deserialized_object = pickle.loads(pickled_object)
    except pickle.UnpicklingError:
        assert False, "new_mediator is not a valid serialized object"
    assert isinstance(deserialized_object, neat.nn.RecurrentNetwork), "new_mediator is not a RecurrentNetwork object"

def test_request_new_mediator_invalid():
    response = client.post("/restart_population")
    response = client.post("/request_new_mediator", json={"mediator_id": "invalid", "session_data": "invalid"})
    assert response.status_code == 422  # Unprocessable Entity

def test_request_new_mediator_missing_fields():
    response = client.post("/restart_population")
    response = client.post("/request_new_mediator", json={"mediator_id": 1})
    assert response.status_code == 422  # Unprocessable Entity
 
def pre_make_individuals(pre_made_individuals : int):
    for i in range(pre_made_individuals):
        logging.info(f"Creating individual {i + 1}")
        time = random.uniform(0, 1000)
        rating = random.randint(1, 5)
        client.post("/user_data", json={"genome_id": i + 1, "time_since_startup": time, "user_rating": rating})
        logging.info(f"Individual {i + 1} posted")

def test_start_evolution_1():
    response = client.post("/restart_population")
    # can only evolve after receiving user data
    pre_make_individuals(1)
    response = client.post("/start_evolution")
    assert response.status_code == 200
    assert response.json() == {"message": "Evolution process started"}

def test_start_evolution_2():
    response = client.post("/restart_population")
    pre_make_individuals(2)
    response = client.post("/start_evolution")
    assert response.status_code == 200
    assert response.json() == {"message": "Evolution process started"}

def test_start_evolution_4():
    new_start = client.post("/restart_population")
    pre_make_individuals(4)
    response = client.post("/start_evolution")
    assert response.status_code == 200
    assert response.json() == {"message": "Evolution process started"}

def test_start_evolution_8():
    new_start = client.post("/restart_population")
    pre_make_individuals(8)
    response = client.post("/start_evolution")
    assert response.status_code == 200
    assert response.json() == {"message": "Evolution process started"}

def test_start_evolution_16():
    new_start = client.post("/restart_population")
    pre_make_individuals(16)
    response = client.post("/start_evolution")
    assert response.status_code == 200
    assert response.json() == {"message": "Evolution process started"}
 
def test_start_evolution_30():
    new_start = client.post("/restart_population")
    pre_make_individuals(30)
    response = client.post("/start_evolution")
    assert response.status_code == 200
    assert response.json() == {"message": "Evolution process started"}