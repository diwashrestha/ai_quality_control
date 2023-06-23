import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from defect_segmentation import UNet, DoubleConvs
from image_utils import image_inference
import random



# Set the layout and title of the Streamlit app
st.set_page_config(layout="wide")
st.title('Steel Defect Detection System 🔍📝')


# File uploader to choose an image
uploaded_file = st.file_uploader("Choose a image file", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    try:
        #img_out = image_inference(uploaded_file, 0.5)
        inf_out_1 = image_inference('test_images/'+ uploaded_file.name, 0.5,1)
        inf_out_4 = image_inference('test_images/'+ uploaded_file.name, 0.5,4)
        inf_out_3 = image_inference('test_images/'+ uploaded_file.name, 0.5,3)

        defect_bool = np.allclose(inf_out_4[1],inf_out_4[0],rtol=1)
        if defect_bool:
            st.subheader("Defect: :green[Not Found]✅")
        else:
            st.subheader('Defect: :red[Found]🚨')


        



        col1, col2 = st.columns(2)
        col1.header('Test Image')
        col1.image(Image.fromarray(inf_out_1[0]))

        col2.header("Test Result")
        col2.image(Image.fromarray(inf_out_4[1]))
    except Exception as e:
        st.error(f"An error occurred: {e}",icon="🚨")

else:
    st.subheader("Try Sample Images")
    
    image_deck = [
        "test_images/m9b1abd245.jpg",
        "test_images/000f6bf48.jpg",
        "test_images/000f6bf48.jpg",
        "test_images/0d2a4e766.jpg",
        "test_images/904677cea.jpg",
        "test_images/14cc1c6b0.jpg",
        "test_images/ma2337ccb3.jpg",
        "test_images/06e6e7a8c.jpg",
        "test_images/n1eeaca7cd.jpg",
        "test_images/n2b6c68337.jpg"
        ]
    
    
    if st.button('Try'):
        # Randomly choose an image from the image deck
        image_number = random.randint(0,8) 

        # Perform image inference for different defect classes
        inf_out_1 = image_inference(image_deck[image_number], 0.5, 1)
        inf_out_3 = image_inference(image_deck[image_number], 0.5, 3)
        inf_out_4 = image_inference(image_deck[image_number], 0.5, 4)

        defect_bool = np.allclose(inf_out_4[1],inf_out_4[0],rtol=1)
        if defect_bool:
            st.subheader("Defect: :green[Not Found]✅")
        else:
            st.subheader('Defect: :red[Found]🚨')

    
        col1, col2 = st.columns(2)
        col1.header('Sample Image')
        col1.image(Image.fromarray(inf_out_1[0]))

        col2.header("Test Result")
        col2.image(Image.fromarray(inf_out_4[1]))
        
    
    col1, col2, col3, col4 = st.columns(4)
    col1.image(image_deck[0],channels="BGR")
    col2.image(image_deck[1],channels="BGR")
    col3.image(image_deck[2],channels="BGR")
    col4.image(image_deck[3],channels="BGR")

