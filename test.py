import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# fixed size use in training
FIXED_IMAGE_SIZE = (128, 128)
UNKNOWN_THRESHOLD = 0.4

# image preprocessing
def preprocessing(image):
    # resize image to fixed size
    image = cv2.resize(image, FIXED_IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0 # do normalization
    return image

def extract_hog_features(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

def extract_multiscale_hog(image):
    hog_128 = extract_hog_features(image)
    small = cv2.resize(image, (64, 64))
    small = cv2.resize(small, FIXED_IMAGE_SIZE)
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
    img_uint8 = (image * 255).astype(np.uint8)
    hist = cv2.calcHist([img_uint8], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_hsv_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
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



# prediction
def predict(dataFilePath, bestModelPath):
    # load model && preprocessing tools
    model = joblib.load(bestModelPath)
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    predictions = []

    image_files = []
    for file in os.listdir(dataFilePath):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_files.append(file)

    # loop over image files
    for image_name in image_files:
        image_path = os.path.join(dataFilePath, image_name)
        image = cv2.imread(image_path)

        if image is None:
            predictions.append(6) # unknown case
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocessing(image) # apply preprocessing

        features = extract_features(image) # apply feature extraction
        features = features.reshape(1, -1)

        # apply same scaling as training
        features = scaler.transform(features)
        features = pca.transform(features)

        # prediction probabilities
        probs = model.predict_proba(features)[0]
        max_prob = np.max(probs)

        if max_prob < UNKNOWN_THRESHOLD:
            predictions.append(6)
        else:
            predictions.append(model.classes_[np.argmax(probs)])

    return predictions
