import os
import cv2
import numpy as np
import joblib
import pandas as pd
from skimage.feature import hog, local_binary_pattern

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (128, 128)
UNKNOWN_THRESHOLD = 0.4
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash", "Unknown"]

# -----------------------------
# Load trained components
# -----------------------------
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# -----------------------------
# Preprocessing and feature extraction
# -----------------------------
def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def extract_hog(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)

def extract_multiscale_hog(img):
    hog1 = extract_hog(img)
    small_img = cv2.resize(img, (64,64))
    small_img = cv2.resize(small_img, IMAGE_SIZE)
    hog2 = extract_hog(small_img)
    return np.concatenate([hog1, hog2])

def extract_lbp(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp_img = local_binary_pattern(gray, 8, 2, method="uniform")
    hist, _ = np.histogram(lbp_img, bins=10, range=(0,10))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_rgb_hist(img):
    img = (img*255).astype(np.uint8)
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hsv_hist(img):
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(img):
    return np.concatenate([extract_multiscale_hog(img), extract_lbp(img), extract_rgb_hist(img), extract_hsv_hist(img)])

# -----------------------------
# Prediction function
# -----------------------------
def predict_folder(folder_path):
    results = []

    # List all images
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".png",".jpeg"))]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            label = "Unknown"
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img)
            features = extract_features(img).reshape(1, -1)
            features = scaler.transform(features)
            features = pca.transform(features)
            probs = model.predict_proba(features)[0]
            max_prob = np.max(probs)
            if max_prob < UNKNOWN_THRESHOLD:
                label = "Unknown"
            else:
                label = CLASSES[np.argmax(probs)]
        results.append({"Image": img_name, "Prediction": label})

    # Save to Excel
    df = pd.DataFrame(results)
    output_path = os.path.join(folder_path, "output.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# -----------------------------
# Run the function
# -----------------------------
predict_folder("sample")
