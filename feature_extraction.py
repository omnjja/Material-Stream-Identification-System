import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image, size=(128, 128)):
    """
    Resize the image and normalize pixel values to [0,1].
    """
    resized = cv2.resize(image, size)
    normalized = resized / 255.0
    return normalized

# -----------------------------
# HOG Feature Extraction
# -----------------------------
def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features

# -----------------------------
# LBP Feature Extraction
# -----------------------------
def extract_lbp_features(image, P=8, R=1):
    """
    Extract Local Binary Pattern (LBP) features from a grayscale image.
    Returns a normalized histogram.
    """
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # normalize
    return hist

# -----------------------------
# Color Histogram
# -----------------------------
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist(
        [image],
        channels=[0, 1, 2],
        mask=None,
        histSize=bins,
        ranges=[0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# -----------------------------
# Combine features
# -----------------------------
def extract_features(image):
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    color_feat = extract_color_histogram((image * 255).astype(np.uint8))
    return np.concatenate([hog_feat, lbp_feat, color_feat])

# -----------------------------
# Extract features from dataset
# -----------------------------
DATASET_PATH = "augmented_dataset"
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

X = []
y = []

for label, cls in enumerate(CLASSES):
    class_folder = os.path.join(DATASET_PATH, cls)
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(".jpg")]

    print(f"Processing class '{cls}' with {len(image_files)} images...")

    for img_file in tqdm(image_files, desc=cls):
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)
        features = extract_features(img)

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature extraction completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Save features for later training
np.save("X_features.npy", X)
np.save("y_labels.npy", y)
print("Features and labels saved as 'X_features.npy' and 'y_labels.npy'")
