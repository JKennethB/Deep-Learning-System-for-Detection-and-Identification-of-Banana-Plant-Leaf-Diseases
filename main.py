import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache the model loading to avoid reloading on every prediction
@st.cache_resource
def get_banana_model():
    return tf.keras.models.load_model("trained_ResNet50_v1.keras",compile=False)

def model_prediction(upload_file):
    model = get_banana_model()
    # Open using PIL to be safes
    img = Image.open(upload_file).resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions[0])

# --- STREAMLIT UI ---
st.header("Banana Leaf Disease Recognition System")

# Use a very specific variable name here
input_file_buffer = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if input_file_buffer is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Preview")
        # Ensure you are calling st.image (the function) 
        # and passing input_file_buffer (the data)
        st.image(input_file_buffer, use_container_width=True)

    with col2:
        st.subheader("Analysis")
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                input_file_buffer.seek(0)
                idx = model_prediction(input_file_buffer)
            
            classes = ["Black Sigatoka", "Healthy", "Panama Disease", "Yellow Sigatoka"]
            st.success(f"Result: {classes[idx]}")