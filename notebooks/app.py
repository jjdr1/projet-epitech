import streamlit as st
from streamlit_image_comparison import image_comparison
import numpy as np


from predictor import Predictor

from PIL import Image
import requests

import pathlib

st.set_page_config(page_title="Image-Comparison Example", layout="centered")

st.title("Veuillez téléverser une photo")
data = st.file_uploader("", type=["jpg", "png"])

toto = Predictor()

if data is not None:
    with open("img1.png", "wb") as f:
        f.write(data.getbuffer())
    st.title("Egalisation de l'exposition de l'image")

    image = toto.load("img1.png")
    equalized = toto.get_equalized_image(image)

    image_comparison(
        img1="img1.png",
        img2=toto.save(equalized, "eq.png"),
        )
    
    st.title("Compression de l'image")

    comp = toto.get_compressed_image(equalized)

    image_comparison(
        img1="img1.png",
        img2=toto.save(comp, "comp1.png"),
    )

    st.title("Prédiction approche")

    pred_y = toto.predict(image.reshape(-1, 4096))
    pred_image = toto.get_image_by_index(pred_y)
    to = toto.save(pred_image, "pred.png")

    image_comparison(
        img1="img1.png",
        img2=to,
        )