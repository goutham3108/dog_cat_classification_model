import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Cat vs. Dog Classifier",
    page_icon="🐾",
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

# --- UI COMPONENTS ---
st.title("Cat vs. Dog Image Classifier")
st.markdown("Upload an image or use your webcam to take a photo, and the model will predict whether it's a cat or a dog.")

# Use session state to track the most recent image
if 'last_image' not in st.session_state:
    st.session_state.last_image = None

col1, col2 = st.columns(2)

with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.last_image = uploaded_file

with col2:
    st.header("Take a Photo")
    camera_image = st.camera_input("Use your webcam")
    if camera_image is not None:
        st.session_state.last_image = camera_image

# --- PREDICTION LOGIC ---
if st.session_state.last_image is not None and model is not None:
    image_to_predict = Image.open(st.session_state.last_image)
    
    st.divider()
    st.subheader("Your Image:")
    st.image(image_to_predict, width=300)

    with st.spinner('Analyzing the image...'):
        processed_image = preprocess_image(image_to_predict)
        prediction = model.predict(processed_image)
        score = float(prediction[0][0])  # ✅ Convert NumPy array to float

    st.subheader("Prediction:")
    if score > 0.5:
        confidence = score * 100
        st.success(f"**This looks like a Dog!** (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - score) * 100
        st.success(f"**This looks like a Cat!** (Confidence: {confidence:.2f}%)")

st.sidebar.markdown("---")
st.sidebar.info(
    "**About this App:**\n"
    "This application uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras, leveraging transfer learning from the MobileNetV2 architecture. It was trained on the Oxford-IIIT Pet Dataset."
)