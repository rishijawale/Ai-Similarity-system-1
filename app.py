import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load saved features
features = pickle.load(open("features.pkl", "rb"))
image_names = pickle.load(open("image_names.pkl", "rb"))

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

def find_similar_images(query_feature, top_k=5):
    similarities = cosine_similarity([query_feature], features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [image_names[i] for i in top_indices]

# UI
st.title("🖼️ AI Image Similarity & Recommendation System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption="Uploaded Image", width=300)

    query_feature = extract_features(query_img)
    similar_images = find_similar_images(query_feature)

    st.subheader("🔍 Similar Images")
    cols = st.columns(len(similar_images))

    for col, img_name in zip(cols, similar_images):
        img_path = os.path.join("images", img_name)
        col.image(img_path, use_column_width=True)

