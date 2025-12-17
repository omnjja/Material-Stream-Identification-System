import cv2
import numpy as np
import joblib
import time
from skimage.feature import hog, local_binary_pattern

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (128, 128)
UNKNOWN_THRESHOLD = 0.5
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

# -----------------------------
# Load trained components
# -----------------------------
print("[INFO] Loading model and scaler...")
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(image, size=IMAGE_SIZE):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image

# -----------------------------
# Feature Extraction (SAME AS TRAINING)
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

def extract_lbp_features(image, P=8, R=2):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

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

def extract_features(image):
    return np.concatenate([
        extract_multiscale_hog(image),
        extract_lbp_features(image),
        extract_color_histogram(image),
        extract_hsv_histogram(image)
    ])

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Camera started. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = preprocess_image(rgb)

    features = extract_features(img).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    probs = model.predict_proba(features_pca)[0]
    max_prob = np.max(probs)

    if max_prob < UNKNOWN_THRESHOLD:
        label = "Unknown"
        confidence = max_prob
    else:
        class_id = np.argmax(probs)
        label = CLASSES[class_id]
        confidence = probs[class_id]

    # FPS calculation
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)

    # Display text
    cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Material Stream Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
