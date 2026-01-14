import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# CONSTANTS
# ===============================
IMAGE_SIZE = 256

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("1model_mobilenet.keras")

model = load_model()

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)               # (H, W, 3)
    image = np.expand_dims(image, axis=0) # (1, H, W, 3)
    return image

# ===============================
# STREAMLIT UI
# ===============================
st.title("Potato Disease Classification")
st.write("Upload a potato leaf image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

