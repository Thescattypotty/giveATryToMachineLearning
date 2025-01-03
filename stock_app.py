
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.title("Prédiction du prix des actions")

model = load_model("stock_price_model.h5")

current_price = st.number_input("Entrez un prix courant", 0.0, 10000.0, 100.0)
if st.button("Prédire"):
    prediction = model.predict(np.array([[current_price]]).reshape(1,1,1))
    st.write("Prix prédit :", prediction[0][0])