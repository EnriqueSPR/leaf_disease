# Import necessary libraries

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from cv2 import imread, resize
from pickle import load
import numpy as np
import matplotlib.pyplot as plt




# Layout
st.set_page_config(
    page_title="disease-estimator",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="auto") 

# Page Layout
col1 = st.sidebar
col2, col3 = st.beta_columns((2,1))

# Logo
image = Image.open('logo.png')
st.image(image, width = 700)

# Title
st.markdown("""## **An App to estimate leaf spot disease levels on cereals 🌾**""")

# Description
st.markdown("""

**Description**: This app was built to predict disease rating, from an input image. \n
The [inception v3](https://keras.io/api/applications/inceptionv3/) model was chosen for this task (Trainable params: 23,869,601):

""")

    
st.markdown("---")

# About

expander_bar = st.beta_expander("About", expanded=False)
expander_bar.markdown("""
* **Python libraries used:** numpy, streamlit, PIL, cv2, pickle, keras (2.4.0).
* **Data**: 144 plant pictures taken from above.
* **Author**: Enrique Alcalde.
---
""")

col1.markdown("""### Instructions:
> 1. Choose one model
> 2. Upload your plant picture""")

models_list = ["Inception v3"]
selected_model = col1.selectbox("Select the Model", models_list)

# component to upload images
uploaded_file = col1.file_uploader(
    "Upload a plant image for disease estimation", type=["jpg", "jpeg", "png"])



# Example picture
st.write("See below an example of a picture for disease score prediction:")
example_img = Image.open('Picture1.png')
st.image(example_img, width = 400)
st.write("""---""")
st.success("See below the disease score")

def prepare_image_inception(im):
    size = (299, 299)
    imag = ImageOps.fit(im, size, Image.ANTIALIAS)
    img_array = np.asarray(imag)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)


@st.cache
def model_load(path):
    model = load_model(path)
    return model

with open("scaler.pickle", 'rb') as pickle_scaler:
        scaler = load(pickle_scaler)


if uploaded_file:
    ima = Image.open(uploaded_file)
    if selected_model == "Inception v3":
        newsize = (250, 250) 
        resized = ima.resize(newsize) 
        shown = ima.resize((350, 250))
        shown_pic = st.image(shown, caption='Your Plant Picture')
        st.write("")
        st.info("Estimating...")

        path = "inception_v3_approach_raw.h5"
        Inception_Resnet = model_load(path)
        prepared_img = prepare_image_inception(ima)

        pred = float(scaler.inverse_transform(Inception_Resnet.predict(prepared_img))[0][0])

       

        results = st.code("The predicted disease score is: {}".format(round(pred,2)))


    
  
