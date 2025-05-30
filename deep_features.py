import numpy as np
import cv2

#tensor flow for deep features
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

#load MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_deep_features(image, mask):
    """
    Apply the mask to the image and extract deep features using MobileNetV2.
    """
    #keep only hand
    masked = cv2.bitwise_and(image, image, mask=mask)

    resized = cv2.resize(masked, (224, 224))

    input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)
    input_tensor = preprocess_input(input_tensor)

    #extract using deep features
    features = model.predict(input_tensor)
    return features.flatten()