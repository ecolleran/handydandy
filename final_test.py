import cv2
import joblib
import numpy as np
from preprocess import preprocess_image, segment_hand
from deep_features import extract_deep_features

#Load model
model = joblib.load("models/svm_model.pkl")

#Map for labels
gesture_labels = {
    "0": "L",
    "1": "Peace",
    "2": "Stop!"
}

#Load static image
image_path = "test_sample.png"
original, blurred = preprocess_image(image_path)
mask = segment_hand(blurred)
features = extract_deep_features(original, mask).reshape(1, -1)

prediction = model.predict(features)[0]
gesture_name = gesture_labels.get(str(prediction), "Unknown")

cv2.putText(original, f"Predicted: {gesture_name}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Test Sample", original)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()