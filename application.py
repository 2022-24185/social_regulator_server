""" This module contains the FastAPI server that will be used to receive user data and start the evolution process. """

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from neuroevolution.server.models import ResponseModel
from neuroevolution.server.config import Config
from neuroevolution.server.models import UserData
from neuroevolution.server.tasks import (process_user_data, reset_population, run_evolution,
                    swap_out_mediator, get_new_mediator, get_experiment_statuses, reset_experiment)
import logging
# pylint: disable=logging-fstring-interpolation
app = FastAPI()

origins = [
    "http://localhost",
    "http://127.0.0.1",
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
    Receive and process user data asynchronously.
    """
    try:
        background_tasks.add_task(process_user_data, user_data)
        return {"message": "Data received successfully"}
    except Exception as e:
        logging.error(f"Error receiving user data: {e}")
        raise HTTPException(status_code=500, detail="Error processing user data")

@app.post("/request_new_mediator")
async def request_new_mediator(background_tasks: BackgroundTasks, user_data: UserData):
    """
    Generate and return a new mediator genome.
    """
    try:
        new_mediator = swap_out_mediator(user_data)
        if new_mediator:
            return JSONResponse(
                status_code=200, 
                content=ResponseModel(phenotype=new_mediator, message="New mediator generated successfully").model_dump()
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate new mediator genome")
    except Exception as e:
        logging.error(f"Error generating new mediator: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.post("/start_evolution")
async def start_evolution(background_tasks: BackgroundTasks):
    """
    Start the evolution process asynchronously.
    """
    try:
        background_tasks.add_task(run_evolution)
        return {"message": "Evolution process started"}
    except Exception as e:
        logging.error(f"Error starting evolution process: {e}")
        raise HTTPException(status_code=500, detail="Failed to start evolution process")

@app.get("/get_first_mediator")
async def get_first_mediator():
    """
    Fetch the first mediator genome.
    """
    try:
        new_mediator = get_new_mediator()
        if new_mediator:
            return JSONResponse(
                status_code=200, 
                content=ResponseModel(phenotype=new_mediator, message="First mediator generated successfully").model_dump()
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate first mediator genome")
    except Exception as e:
        logging.error(f"Error fetching first mediator: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.post("/restart_population")
async def restart_population(background_tasks: BackgroundTasks):
    """
    Restart the population asynchronously.
    """
    try:
        background_tasks.add_task(reset_population)
        return {"message": "Population reset successfully"}
    except Exception as e:
        logging.error(f"Error restarting population: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset population")

@app.get("/server_status")
async def task_get_server_status():
    """
    Get the current server status, including experiment data.
    """
    try:
        status = get_experiment_statuses()
        return {"message": "Server status retrieved successfully", "status": status}
    except Exception as e:
        logging.error(f"Error retrieving server status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve server status")

@app.post("/reset_experiment")
async def task_reset_experiment(background_tasks: BackgroundTasks):
    """
    Reset the experiment asynchronously.
    """
    try:
        background_tasks.add_task(reset_experiment)
        return {"message": "Experiment reset started"}
    except Exception as e:
        logging.error(f"Error starting experiment reset: {e}")
        raise HTTPException(status_code=500, detail="Failed to start experiment reset")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.SERVER_HOST, port=Config.SERVER_PORT)