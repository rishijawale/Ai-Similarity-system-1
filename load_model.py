from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50 model
model = ResNet50(
    weights='imagenet',     # already trained
    include_top=False,      # remove classification part
    pooling='avg'           # convert image to feature vector
)

print("✅ ResNet50 model loaded successfully")
