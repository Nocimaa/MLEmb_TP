import fastapi
import joblib
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

from pydantic import BaseModel

vectorizer, clf = joblib.load("multi_class_model.joblib")

app = fastapi.FastAPI()



class Payload(BaseModel):
    string: str



@app.get("/predict")
def predict():
    return {"y_pred": 2}

@app.post("/predict")
def predict(payload: Payload):

    X_vect = vectorizer.transform([payload.string])
    pred = int(clf.predict(X_vect)[0]) 
    return {"rating": pred}
