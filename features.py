import cv2
import numpy as np

def extract_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found. Returning zeros.")
        return np.zeros(12)

    largest_contour = max(contours, key=cv2.contourArea)

    #Hu Moments (7 features)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    #Convex Hull and Convexity Defects
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, hull)
    num_defects = defects.shape[0] if defects is not None else 0

    #Contour Area and Aspect Ratio
    area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h

    #Convex Hull Area
    hull_points = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull_points)

    #Extent (Contour area / bounding box area)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0

    #12 features
    features = np.hstack([hu_moments, num_defects, area, aspect_ratio, hull_area, extent])
    return features