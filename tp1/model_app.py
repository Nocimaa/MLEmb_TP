import streamlit
import joblib

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

@streamlit.cache_data
def load_model():
    model = joblib.load("regression.joblib")
    return model

model = load_model()
streamlit.title("House Price Prediction")

size = streamlit.number_input("Size (in square meters)", min_value=10, max_value=1000, value=50)
nb_rooms = streamlit.number_input("Number of rooms", min_value=1, max_value=30, value=3)
garden = streamlit.checkbox("Has garden")

if streamlit.button("Predict Price"):
    price = model.predict([[size, nb_rooms, garden]])
    streamlit.write(f"Predicted Price: {price[0]:.2f} euros")