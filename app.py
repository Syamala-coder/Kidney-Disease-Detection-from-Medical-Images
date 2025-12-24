import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image


import streamlit as st
import base64


def set_bg_image(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# CALL background function
set_bg_image(r"background_image.jpeg")







# -------------------------------
# Load trained Decision Tree model
# -------------------------------
with open("dt.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Kidney Disease Detection From Medical Images")
st.write("Upload a kidney image to predict the disease")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=320)


    st.warning(
        "⚠️ Please upload only kidney-related medical images (CT/MRI). "
        "Uploading other images may result in incorrect predictions."
    )


    # -------------------------------
    # PREPROCESSING (EXACT SAME AS TRAINING)
    # -------------------------------
    image = np.array(image)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 64x64
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    # Normalize
    image = image.astype("float32") / 255.0

    # Flatten
    features = image.flatten().reshape(1, -1)   # (1, 4096)

  


        
    confirm = st.checkbox("I confirm that the uploaded image is a kidney medical image")

if st.button("Predict"):
    if not confirm:
        st.error("Please confirm that the uploaded image is a kidney-related medical image.")
    else:
        prediction = model.predict(features)
        st.markdown(
            f"""
            <div style="color:white; font-size:22px; font-weight:600;">
                Predicted Class: {prediction[0]}
            </div>
            """,
            unsafe_allow_html=True
        )