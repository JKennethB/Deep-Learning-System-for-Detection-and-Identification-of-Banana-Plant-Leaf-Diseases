import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache the model loading to avoid reloading on every prediction
@st.cache_resource
def resnet_model():
    return tf.keras.models.load_model("trained_ResNet50.keras",compile=False)

def model_prediction(upload_file):
    model = resnet_model()
    # Open using PIL to be safe
    img = Image.open(upload_file).resize((512, 512))
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
            
            classes = ["Black Sigatoka", "Healthy Leaf", "Panama Disease", "Yellow Sigatoka"]
            result = classes[idx]
            st.success(f"Result: {result}")

            solutions = {
                "Black Sigatoka": [
                    "Remove and destroy infected leaves to reduce the spread of the disease.",
                    "Ensure proper spacing between plants to improve air circulation and reduce humidity.",
                    "Consider applying fungicides as recommended by agricultural experts to control the disease."
                ],
                "Yellow Sigatoka": [
                    "Remove and destroy infected leaves to prevent the spread of the disease.",
                    "Improve soil drainage to reduce moisture levels that favor the disease.",
                    "Consider applying fungicides as recommended by agricultural experts to control the disease."
                ],
                "Panama Disease": [
                    "Remove and destroy infected plants to prevent the spread of the disease.",
                    "Improve soil drainage to reduce moisture levels that favor the disease.",
                    "Consider crop rotation and planting resistant banana varieties to manage the disease."
                ],
                "Healthy Leaf": [
                    "The leaf is healthy. No specific actions are needed."
                ]
            }

            with st.expander("View Recommended Solutions"):
                for solution in solutions[result]:
                    st.write(f"- {solution}")