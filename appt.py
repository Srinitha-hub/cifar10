import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0
y_test = to_categorical(y_test, 10)

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('cnn_model.h5')

model = load_model()

# Class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title('CIFAR-10 Image Classification')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = image.load_img(uploaded_file, target_size=(32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    if st.button('Predict'):
        img_array = image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        st.write(f"Predicted Class: {class_names[predicted_class]}")
