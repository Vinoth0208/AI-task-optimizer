import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import date

model = tf.keras.models.load_model("submission_model.h5")
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

job_assignment = {
    'Anger': 'Handle complaints or negotiation tasks',
    'Disgust': 'Quality inspection or audit tasks',
    'Fear': 'Review risk or security protocols',
    'Happy': 'Customer engagement or team motivation',
    'Neutral': 'Focus on data analysis or regular operations',
    'Sadness': 'Work on solo tasks like writing or reviewing',
    'Surprise': 'Creative brainstorming or innovation tasks'
}

def predict_emotion(img):
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    predictions = model.predict(img_expanded)
    class_idx = np.argmax(predictions)
    return CLASS_LABELS[class_idx]

st.set_page_config(layout="centered")
st.title("🧠 Emotion-Based Daily Job Assignment")
st.markdown(f"📅 **Today:** {date.today().strftime('%A, %d %B %Y')}")

mode = st.radio("Choose Input Method", ["📁 Upload Image", "📷 Use Webcam", "🎯 Choose Emotion"])

if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image_np, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict Emotion"):
            predicted_emotion = predict_emotion(image_np)
            assigned_job = job_assignment[predicted_emotion]
            st.success(f"Detected Emotion: **{predicted_emotion}**")
            st.info(f"🎯 **Assigned Task:** {assigned_job}")

elif mode == "📷 Use Webcam":
    picture = st.camera_input("Take a photo")

    if picture is not None:
        image = Image.open(picture).convert("RGB")
        image_np = np.array(image)
        st.image(image_np, caption="Captured Image", use_container_width=True)
        if st.button("Predict Emotion"):
            predicted_emotion = predict_emotion(image_np)
            assigned_job = job_assignment[predicted_emotion]
            st.success(f"Detected Emotion: **{predicted_emotion}**")
            st.info(f"🎯 **Assigned Task:** {assigned_job}")

elif mode == "🎯 Choose Emotion":
    selected_emotion = st.selectbox("Pick Employee's Emotion", CLASS_LABELS)
    if selected_emotion:
        assigned_job = job_assignment[selected_emotion]
        st.success(f"Selected Emotion: **{selected_emotion}**")
        st.info(f"🎯 **Assigned Task:** {assigned_job}")



