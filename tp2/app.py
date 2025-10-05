import fastapi
import mlflow
import mlflow.sklearn
import warnings
import numpy as np
warnings.filterwarnings(action='ignore', category=UserWarning)
import pandas as pd
from pydantic import BaseModel

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


model_name = "InsuranceModel"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)


app = fastapi.FastAPI()



class Insurance(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class Version(BaseModel):
    model_version: str

import sklearn.preprocessing as preprocessing


@app.post("/predict")
def predict(insurance: Insurance):

    print(insurance.model_dump())
    df = pd.DataFrame(insurance.model_dump(), index=[0])


    # data = np.asarray([[insurance.age, insurance.sex, insurance.bmi, insurance.children, insurance.smoker, insurance.region]], dtype="object")
    y_pred = model.predict(df)
    return {"prediction": y_pred[0]}

@app.post("/update-model endpoint")
def update_model(version: Version):
    global model
    model_uri = f"models:/{model_name}/{version.model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return {"message": "Model updated successfully"}
