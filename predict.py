import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('deepfake_detection_model.h5')

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (96, 96))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image.astype(np.float32))
    return np.expand_dims(image, axis=0)

# Predict if the image is fake or real
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image, verbose=0)
    class_label = np.argmax(prediction, axis=1)[0]
    return "Fake" if class_label == 0 else "Real"

if __name__ == "__main__":
    image_path = "real_and_fake_face_detection/real_and_fake_face/training_real/real_00001.jpg"
    result = predict_image(image_path)
    print(f"The image is {result}")
