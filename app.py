import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set class names
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Load model from selected option
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join("SavedModels", f"{model_name}.h5")
    return tf.keras.models.load_model(model_path)

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🫘 Bean Leaf Disease Classifier")
st.markdown("Upload a bean leaf image and select a model to classify it.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model selection
model_option = st.selectbox("Choose a model", ['MobileNetV2', 'EfficientNet', 'ResNet50', 'VGG16'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        model = load_model(model_option)
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"🧠 Predicted: **{predicted_class}** ({confidence*100:.2f}% confidence)")
