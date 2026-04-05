import streamlit as st
import streamlit_analytics2 as st_analytics
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input


# Helper function to log predictions globally
def log_to_global_history(file_name, prediction, confidence, inf_time):
    history_file = "global_history.csv"
    new_data = pd.DataFrame([[time.strftime("%Y-%m-%d %H:%M:%S"), file_name, prediction, confidence, inf_time]], 
                            columns=["Timestamp", "File Name", "Prediction", "Confidence", "Inference Time"])
    
    # If file exists, append; otherwise, create it
    if os.path.exists(history_file):
        new_data.to_csv(history_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(history_file, index=False)

# Cache the model loading to avoid reloading on every prediction
@st.cache_resource
def resnet_model():
    return tf.keras.models.load_model("trained_ResNet50.keras",compile=False)

def model_prediction(upload_file):
    model = resnet_model()
    # Open using PIL to be safe
    img = Image.open(upload_file).convert("RGB").resize((512, 512))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    input_arr = preprocess_input(input_arr)  # Preprocess for ResNet50

    start_time = time.time()
    predictions = model.predict(input_arr)
    end_time = time.time()

    inf_time = end_time - start_time
    idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100  # Get confidence percentage
    return idx, confidence, inf_time

# --- STREAMLIT UI ---
# Centered Header
st.markdown("<h1 style='text-align: center;'>Banana Leaf Disease Recognition</h1>", unsafe_allow_html=True)

# Centered Subheader or Normal Text
st.markdown("<p style='text-align: center;'>Upload an image to start the analysis</p>", unsafe_allow_html=True)

# Use a very specific variable name here
input_file_buffer = st.file_uploader(" ", type=["jpg", "png", "jpeg"])

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
                idx, confidence, inf_time = model_prediction(input_file_buffer)
            
            classes = ["Black Sigatoka", "Healthy Leaf", "Panama Disease", "Yellow Sigatoka"]
            result = classes[idx]

            log_to_global_history(input_file_buffer.name, result, f"{confidence-10:.2f}%", f"{inf_time:.2f} seconds")

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

# History Section

st.divider()

with st.expander("View Prediction History"):
    st.write("Your recent activity using this system:")

    if os.path.exists("global_history.csv"):
        # Read the CSV and show the last 10 rows
        df = pd.read_csv("global_history.csv")
                        
        # Show as a clean table (newest first)
        st.table(df.tail(10).iloc[::-1])
        
        col1, col2 = st.columns(2)
        with col1:
            # Optional: Download button for the full history
            st.download_button(
                label="Download Full Global History",
                data=df.to_csv(index=False),
                file_name="global_banana_history.csv",
                mime="text/csv",
        )
        
        with col2:
            # Optional: Clear history button
            if st.button("Clear Global History"):
                os.remove("global_history.csv")
    else:
        st.info("Nothing recorded yet.")