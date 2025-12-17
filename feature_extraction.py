import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (128, 128)
DATASET_PATH = "augmented_dataset"
NEW_PATH ="dataset"
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image, size=IMAGE_SIZE):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image

# -----------------------------
# HOG Feature Extraction
# -----------------------------
def extract_hog_features(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

def extract_multiscale_hog(image):
    hog_128 = extract_hog_features(image)

    small = cv2.resize(image, (64, 64))
    small = cv2.resize(small, IMAGE_SIZE)
    hog_64 = extract_hog_features(small)

    return np.concatenate([hog_128, hog_64])

# -----------------------------
# LBP Feature Extraction
# -----------------------------
def extract_lbp_features(image, P=8, R=2):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")

    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist

# -----------------------------
# Color Histograms
# -----------------------------
def extract_color_histogram(image, bins=(8, 8, 8)):
    image_uint8 = (image * 255).astype(np.uint8)
    hist = cv2.calcHist(
        [image_uint8],
        [0, 1, 2],
        None,
        bins,
        [0, 256, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hsv_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

# -----------------------------
# Combine Features
# -----------------------------
def extract_features(image):
    hog_feat = extract_multiscale_hog(image)
    lbp_feat = extract_lbp_features(image)
    rgb_feat = extract_color_histogram(image)
    hsv_feat = extract_hsv_histogram(image)

    return np.concatenate([
        hog_feat,
        lbp_feat,
        rgb_feat,
        hsv_feat
    ])

# -----------------------------
# Dataset Feature Extraction
# -----------------------------
X = []
y = []

print("\nStarting feature extraction...\n")

for label, cls in enumerate(CLASSES):
    class_dir = os.path.join(DATASET_PATH, cls)
    image_files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Processing '{cls}' ({len(image_files)} images)")

    for img_name in tqdm(image_files, desc=cls):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)

        features = extract_features(img)
        X.append(features)
        y.append(label)


##############################################################
for label, cls in enumerate(CLASSES):
    class_dir = os.path.join(NEW_PATH, cls)
    image_files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Processing '{cls}' ({len(image_files)} images)")

    for img_name in tqdm(image_files, desc=cls):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)

        features = extract_features(img)
        X.append(features)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("\nFeature extraction completed!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# Save Features
# -----------------------------
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\nSaved:")
print(" - X_features.npy")
print(" - y_labels.npy")