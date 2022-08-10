import streamlit as st
from PIL import Image
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

st.title("Tea leaf diseas detection")
st.header("Helopeltis")

file_uploaded = st.file_uploader("Choose the file")
if file_uploaded is not None:
    image = Image.open(file_uploaded)
    st.write('Classifying')

predict_result = st.button('Predict')

class_name = ['Healthy_leaf' , 'Sick_leaf']

def predict_class(image):
    classifier_model = keras.models.load_model('FineTuning')
    test_image = image.resize((256,256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image = tf.expand_dims(test_image, 0)
    predictions = classifier_model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    image_class = class_name[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if predict_result:  
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    result = predict_class(image)
    st.write(result)
    st.pyplot(figure)
else:
    st.write('')