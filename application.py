""" This module contains the FastAPI server that will be used to receive user data and start the evolution process. """

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from neuroevolution.server.models import ResponseModel

from neuroevolution.server.config import Config
from neuroevolution.server.models import UserData
from neuroevolution.server.tasks import (process_user_data, reset_population, run_evolution,
                    swap_out_mediator, get_new_mediator)
from neuroevolution.lab.lab import Lab

app = FastAPI()

origins = [
    "http://localhost",
    "https://dev.d26v103nvsfj6o.amplifyapp.com/",
]

# Expand localhost to allow any port
expanded_origins = [origin + ":*" for origin in origins if "localhost" in origin]

# Combine expanded localhost and specific domain
all_origins = expanded_origins + ["https://your-amplify-app-domain.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/user_data")
async def receive_user_data(user_data: UserData, background_tasks: BackgroundTasks):
    """
    Receive user data sent by the client.
    """
    background_tasks.add_task(process_user_data, user_data)
    return {"message": "Data received successfully"}

@app.post("/request_new_mediator")
async def request_new_mediator(background_tasks: BackgroundTasks, user_data: UserData):
    """
    Request a new mediator genome to be generated.
    """
    new_mediator = swap_out_mediator(user_data)
    if new_mediator:
        return JSONResponse(status_code=200, content=ResponseModel(phenotype=new_mediator, message="New mediator generated successfully").model_dump())
    else:
        raise HTTPException(status_code=500, detail="Failed to generate new mediator genome")

@app.post("/start_evolution")
async def start_evolution(background_tasks: BackgroundTasks):
    """
    Start the evolution process.
    """
    background_tasks.add_task(run_evolution)
    return {"message": "Evolution process started"}

@app.get("/get_first_mediator")
async def get_first_mediator():
    """
    Get the first mediator genome.
    """
    new_mediator = get_new_mediator()
    if new_mediator:
        return JSONResponse(status_code=200, content=ResponseModel(phenotype=new_mediator, message="First mediator generated successfully").model_dump())
    else:
        raise HTTPException(status_code=500, detail="Failed to generate first mediator genome")

@app.post("/restart_population")
async def restart_population(background_tasks: BackgroundTasks):
    """
    Restart the population.
    """
    background_tasks.add_task(reset_population)
    return {"message": "Population reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.SERVER_HOST, port=Config.SERVER_PORT)