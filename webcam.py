import cv2
import joblib
import numpy as np
from preprocess import preprocess_frame, segment_hand
from features import extract_features

# Load trained classifier
model = joblib.load("models/svm_model.pkl")

# Map class labels to gesture names
gesture_labels = {
    "0": "L",
    "1": "Peace",
    "2": "Stop!"
}

# Start webcam
cam = cv2.VideoCapture(0)

while (True):
    retval, frame = cam.read()

    #cv2.imshow("Live WebCam", img)
    resized_frame, blurred_frame = preprocess_frame(frame)

    # Segment hand and extract features
    mask = segment_hand(blurred_frame)
    features = extract_features(mask).reshape(1, -1)

    # Predict gesture
    prediction = model.predict(features)[0]
    gesture_name = gesture_labels.get(str(prediction), "Unknown")

    # Display the prediction on the frame
    cv2.putText(resized_frame, f"Gesture: {gesture_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Hand Gesture Recognition", resized_frame)
    cv2.imshow("Mask", mask)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
