import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Cat vs. Dog Classifier",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model('cat_dog_classifier.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'cat_dog_classifier.keras' file is in the same directory as this script.")
        st.error("If you haven't trained the model yet, please run 'python train_model.py' first.")
        return None

model = load_model()

# --- IMAGE PREPROCESSING ---
def preprocess_image(image):
    """Preprocesses the image for model prediction."""
    img = image.resize((160, 160))
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- SESSION STATE ---
if 'last_images' not in st.session_state:
    st.session_state.last_images = []

# --- UI COMPONENTS ---
st.title("Cat vs. Dog Image Classifier")
st.markdown("Upload one or more images, or use your webcam to take a photo — the model will predict whether each is a cat or a dog.")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Images")
    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded_files:
        st.session_state.last_images = uploaded_files
    elif uploaded_files == []:
        st.session_state.last_images = []

with col2:
    st.header("Take a Photo")
    camera_image = st.camera_input("Use your webcam", key="camera_input")
    if camera_image is not None:
        st.session_state.last_images = [camera_image]
    elif camera_image is None and st.session_state.last_images:
        # If user cleared camera photo (and previous was a camera photo), clear results
        if len(st.session_state.last_images) == 1 and not hasattr(st.session_state.last_images[0], "name"):
            st.session_state.last_images = []

# --- PREDICTION LOGIC ---
if model is not None and st.session_state.last_images:
    st.divider()
    for img_file in st.session_state.last_images:
        image_to_predict = Image.open(img_file)
        st.subheader(f"Prediction for: {getattr(img_file, 'name', 'Camera Photo')}")
        st.image(image_to_predict, width=300)

        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image_to_predict)
            prediction = model.predict(processed_image)
            score = float(prediction[0][0])

        if score > 0.5:
            confidence = score * 100
            st.success(f"Dog detected (Confidence: {confidence:.2f}%)")
        else:
            confidence = (1 - score) * 100
            st.success(f"Cat detected (Confidence: {confidence:.2f}%)")

st.sidebar.markdown("---")
st.sidebar.info(
    "**About this App:**\n"
    "This application uses a CNN built with TensorFlow/Keras, leveraging transfer learning from MobileNetV2. "
    "It was trained on the Oxford-IIIT Pet Dataset."
)
