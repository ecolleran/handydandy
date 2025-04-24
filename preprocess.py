import cv2
import numpy as np

def preprocess_image(image_path):
    """Load, resize, and blur image from file path for training."""
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (300, 300))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return resized, blurred

def preprocess_frame(frame):
    """Resize and blur frame directly from webcam for preprocessing."""
    resized = cv2.resize(frame, (300, 300))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return resized, blurred

def segment_hand(image):
    """Segment hand using HSV skin detection and adaptive thresholding."""
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV Skin color range (tune if necessary)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Adaptive Thresholding to handle varying lighting conditions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Combine both masks to robustly isolate the hand
    combined_mask = cv2.bitwise_and(skin_mask, adaptive_mask)

    # Morphological operations (noise cleanup)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(combined_mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # Extract largest contour (hand area)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return refined_mask
    else:
        return mask
