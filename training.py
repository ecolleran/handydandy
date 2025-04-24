import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_image, segment_hand
from features import extract_features

def load_dataset(data_dir):
    X, y = [], []
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            filepath = os.path.join(label_path, filename)
            _, hsv = preprocess_image(filepath)
            mask = segment_hand(hsv)
            features = extract_features(mask)
            X.append(features)
            y.append(label_folder)
    return np.array(X), np.array(y)

def train_and_evaluate(data_path="data"):
    X, y = load_dataset(data_path)
    print(f"Total samples loaded: {len(X)}")
    print(f"Feature vector shape: {X.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    val_acc = accuracy_score(y_val, y_val_pred) * 100

    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("Classification Report:\n", classification_report(y_val, y_val_pred))

    # Save model
    joblib.dump(clf, "models/svm_model.pkl")


if __name__ == "__main__":
    train_and_evaluate("data")
