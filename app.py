import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Class names must match your training order
class_names = ['Glioma', 'Meningioma', 'No Tumour', 'Pituitary']

st.title("🧬 Multi-Class Tumour Image Classifier")
st.write("Upload a medical image to classify it into one of four categories of Tumour.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tumour_multiclass_model.h5')

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_array = np.array(image) / 255.0
    image_array = np.reshape(image_array, (1, 150, 150, 3))

    prediction = model.predict(image_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[predicted_index]

    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Confidence: {confidence:.2%}")

    # Optional: Show all class probabilities
    st.write("Class Probabilities:")
    for i, prob in enumerate(prediction):
        st.write(f"**{class_names[i]}**: {prob:.2%}")
