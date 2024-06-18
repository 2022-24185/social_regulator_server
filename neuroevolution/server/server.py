from fastapi import FastAPI, BackgroundTasks, HTTPException
from .models import UserData
from .tasks import process_user_data, swap_out_mediator, run_evolution, reset_population
from .config import Config
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class ResponseModel(BaseModel):
    new_mediator: str
    message: str

app = FastAPI()

@app.post("/user_data")
async def receive_user_data(user_data: UserData, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_user_data, user_data)
    return {"message": "Data received successfully"}

@app.post("/request_new_mediator")
async def request_new_mediator(user_data: UserData, background_tasks: BackgroundTasks):
    new_mediator = swap_out_mediator(user_data)
    if new_mediator:
        return JSONResponse(status_code=200, content=ResponseModel(new_mediator=new_mediator, message="New mediator generated successfully").model_dump())
    else:
        raise HTTPException(status_code=500, detail="Failed to generate new mediator genome")

@app.post("/start_evolution")
async def start_evolution(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_evolution)
    return {"message": "Evolution process started"}

@app.post("/restart_population")
async def restart_population(background_tasks: BackgroundTasks):
    background_tasks.add_task(reset_population)
    return {"message": "Population reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.SERVER_HOST, port=Config.SERVER_PORT)