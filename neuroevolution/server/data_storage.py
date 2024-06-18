# neuroevolution/pureples/shared/data_storage.py
from typing import TYPE_CHECKING
import csv
from neuroevolution.server.models import UserData

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

class SessionData:
    def __init__(self, filename):
        self.filename = filename

    def store_session_data(self, data: 'UserData'):
        # Check if file exists
        try:
            with open(self.filename, 'x', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(UserData.__annotations__.keys())
                writer.writerow(data.model_dump().values())  # Write values as data
        except FileExistsError:
            # If file exists, append data
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data.model_dump().values())  # Write values as data