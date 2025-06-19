import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# 1️⃣ Constants
MODEL_PATH = "tumour_multiclass_model.h5"
FILE_ID = "1YLDh4r1a1JBf3WQUUWI37zOwUMStWMSB"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# 2️⃣ Download model if not already present
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# 3️⃣ Load the model
model = download_model()

# 4️⃣ Class names (change as per your dataset)
class_names = ['Glioma', 'Meningioma', 'No tumour', 'Pituitary']  # Example class names

# 5️⃣ Streamlit App Interface
st.title("🧠 Tumour Type Classifier (4-Class CNN)")
st.write("Upload an image of a tumour scan to classify it into one of the following:")
st.write(", ".join(f"`{cls}`" for cls in class_names))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.reshape(img_array, (1, 150, 150, 3))

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[predicted_index]

    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Confidence: {confidence:.2%}")

    st.write("Class Probabilities:")
    for i, prob in enumerate(prediction):
        st.write(f"**{class_names[i]}**: {prob:.2%}")
