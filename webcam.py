import cv2
import joblib
import numpy as np

#imports from other project files
from preprocess import preprocess_frame, segment_hand
from deep_features import extract_deep_features

#Load classifier
model = joblib.load("models/svm_model.pkl")

#Map for gesture names
gesture_labels = {
    "0": "L",
    "1": "Peace",
    "2": "Stop!"
}

cam = cv2.VideoCapture(0)

while (True):
    retval, frame = cam.read()

    #cv2.imshow("Live WebCam", img)
    resized_frame, blurred_frame = preprocess_frame(frame)

    #Segment & extract features
    mask = segment_hand(blurred_frame)
    features = extract_deep_features(resized_frame, mask).reshape(1, -1)

    prediction = model.predict(features)[0]
    gesture_name = gesture_labels.get(str(prediction), "Unknown")

    #Label gesture on frame
    cv2.putText(resized_frame, f"Gesture: {gesture_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", resized_frame)
    cv2.imshow("Mask", mask)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
