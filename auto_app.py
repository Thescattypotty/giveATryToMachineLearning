import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Classification d'images automobiles")

# Charger le modèle VGG16 pour l'extraction des caractéristiques
vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Charger le modèle final
model = tf.keras.models.load_model("final_auto_model.h5")
classes = ["audi", "lomborghini", "mercedes"]  # Adapter selon vos besoins

uploaded_file = st.file_uploader("Choisissez une image d'automobile")
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Auto à classer", use_container_width=True)
    if st.button("Classer"):
        img_array = np.array(image)/255.0
        img_array = img_array.reshape((1,224,224,3))
        features = vgg_model.predict(img_array)  # Extraction des caractéristiques avec VGG16
        prediction = model.predict(features)
        st.write("Marque prédite :", classes[np.argmax(prediction[0])])