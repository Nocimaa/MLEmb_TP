import fastapi
import joblib
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

from pydantic import BaseModel

model = joblib.load("regression.joblib")

app = fastapi.FastAPI()



class House(BaseModel):
    size: float
    nb_rooms: int
    garden: bool



@app.get("/predict")
def predict():
    return {"y_pred": 2}

@app.post("/predict")
def predict(house: House):
    y_pred = model.predict([[house.size, house.nb_rooms, house.garden]])
    return {"y_pred": y_pred[0]}
