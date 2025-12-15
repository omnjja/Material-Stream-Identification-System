import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

# -------- LBP parameters --------
LBP_P = 8  # number of neighbors
LBP_R = 1  # radius
LBP_METHOD = 'uniform'


def extract_lbp(image_gray):
    # Compute LBP
    lbp = local_binary_pattern(image_gray, LBP_P, LBP_R, LBP_METHOD)

    # Histogram
    n_bins = LBP_P + 2  # for uniform patterns
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


# -------- HOG parameters --------
def extract_hog(image_gray):
    hog_features = hog(image_gray,
                       orientations=9,
                       pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       transform_sqrt=True)
    return hog_features


# -------- Combined LBP + HOG --------
def extract_features(image_path):
    # read image
    img = cv2.imread(image_path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize for consistency
    gray = cv2.resize(gray, (128, 128))

    # extract LBP & HOG
    lbp_features = extract_lbp(gray)
    hog_features = extract_hog(gray)

    # combine
    combined = np.concatenate([lbp_features, hog_features])

    return combined
