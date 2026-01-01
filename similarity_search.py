import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load saved data
features = pickle.load(open("features.pkl", "rb"))
image_names = pickle.load(open("image_names.pkl", "rb"))

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

def find_similar_images(query_image_path, top_k=5):
    query_feature = extract_features(query_image_path)
    similarities = cosine_similarity([query_feature], features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(image_names[i], similarities[i]) for i in top_indices]

# Test with any image path
results = find_similar_images("images/" + image_names[0])

print("🔍 Similar Images Found:")
for img, score in results:
    print(img, "→ similarity:", round(score, 3))
