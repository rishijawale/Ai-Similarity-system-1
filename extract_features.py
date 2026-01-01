import os
import numpy as np
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load pretrained model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

image_folder = "images"
features = []
image_names = []

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    try:
        feature = extract_features(img_path)
        features.append(feature)
        image_names.append(img_name)
        print(f"✔ Processed: {img_name}")
    except:
        print(f"❌ Skipped: {img_name}")

# Save features
pickle.dump(features, open("features.pkl", "wb"))
pickle.dump(image_names, open("image_names.pkl", "wb"))

print("🎉 Feature extraction completed successfully")
