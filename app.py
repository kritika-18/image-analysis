import streamlit as st
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.image('My-Image.jpg')
st.header('Pneumonia Detection')
st.text('Please upload your X-ray image for detection!')
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)

def loading():
    fp="cnn_pneu_vamp_model.h5"
    model_loader=load_model(fp)
    return model_loader

model=loading()

temp=st.file_uploader("Browse files")
buffer=temp
temp_file=NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))

if buffer is None:
    st.text("Oops! That does not look like an image. Please upload an image file.")
else:
    img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

    # Preprocessing the image
    pp_img = image.img_to_array(img)
    pp_img = pp_img/255
    pp_img = np.expand_dims(pp_img, axis=0)

    #predict
    preds= model.predict(pp_img)
    if preds>= 0.5:
        out = ('There is a {:.2%} percent surety that this is a Pneumonia case'.format(preds[0][0]))

    else:
        out = ('There is a {:.2%} percent surety that this is a Normal case'.format(1-preds[0][0]))

    st.success(out)

    image = Image.open(temp)
    st.image(image,use_column_width=True)

st.write('by Kritika Chauhan 😁')
sen='Nowadays, artificial intelligence is being an important part in every field. Even in the medical field, many medical analysis are being done using the latest technologies. Disease detection is one of those applications.'
sen2='This application of pneumonia detection has been made using the deep learning technique. For this, CNN algorithm has been used.'

st.sidebar.write(sen)
st.sidebar.write(sen2)
st.sidebar.write("How does CNN works?")
st.sidebar.image('CNN1.jpg')
st.sidebar.write('CNN, stands for Convolutional Neural Networks, is used to detect patterns in any image and based on those patterns it classifies the images into different categories.')
st.sidebar.write(" CNNs use image recognition and classification in order to detect objects, recognize faces, etc. They are made up of neurons with learnable weights and biases. Each specific neuron receives numerous inputs and then takes a weighted sum over them, where it passes it through an activation function and responds back with an output.")