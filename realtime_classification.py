import cv2
import numpy as np
import joblib
import time
from skimage.feature import hog, local_binary_pattern

# ------------------------------------
# Basic settings
# ------------------------------------
img_size = (128, 128)          # el sora kolha bnt7wl lel size da
unknown_limit = 0.5            # law el confidence a2al mn kda bn3tbrha unknown
labels_names = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

# ------------------------------------
# Load trained stuff
# ------------------------------------
print("[INFO] Loading model and tools...")
# hena bn-load el model elly et3lm abl kda
svm_model = joblib.load("svm_model.pkl")

# scaler da elly kan by3ml standardization lel features
scale_tool = joblib.load("scaler.pkl")

# PCA tool 3shan n2ll el dimensions
pca_tool = joblib.load("pca.pkl")

# ------------------------------------
# Image preprocessing
# ------------------------------------
def prepare_image(img):
    """
    Resize + normalize image
    bn3ml resize lel sora 3shan kol el images tb2a nafs el size
    w bn2sm 3la 255 3shan el values tb2a mn 0 l 1
    """
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return img

# ------------------------------------
# Feature extraction functions
# ------------------------------------
def hog_features(img):
    """
    extract HOG features
    HOG bymsk el edges w el shapes
    da mohem 3shan nfr2 ben el materials
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

def hog_multi_scale(img):
    """
    HOG mn aktar mn scale
    mara 3la el sora el kbera
    w mara 3la version asghar
    3shan nzwd el details
    """
    f1 = hog_features(img)

    # bn3ml resize asghar
    small_img = cv2.resize(img, (64, 64))
    small_img = cv2.resize(small_img, img_size)

    f2 = hog_features(small_img)

    return np.concatenate((f1, f2))

def lbp_features(img):
    """
    LBP features
    el LBP byfocus 3la el texture
    zay el smooth aw rough
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp_img = local_binary_pattern(gray, 8, 2, method="uniform")

    hist, _ = np.histogram(lbp_img, bins=10, range=(0, 10))
    hist = hist.astype(np.float32)

    # bn-normalize el histogram
    hist /= (hist.sum() + 1e-6)
    return hist

def rgb_hist(img):
    """
    RGB color histogram
    da by2ool el sora feha anhy alwan aktar
    """
    img = (img * 255).astype(np.uint8)
    h = cv2.calcHist(
        [img],
        [0, 1, 2],
        None,
        (8, 8, 8),
        [0, 256, 0, 256, 0, 256]
    )
    h = h.flatten().astype(np.float32)
    h /= (h.sum() + 1e-6)
    return h

def hsv_hist(img):
    """
    HSV histogram
    mohem 3shan lighting
    3shan el nor law etghyar ma ybozsh el result
    """
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        (8, 8, 8),
        [0, 180, 0, 256, 0, 256]
    )
    h = h.flatten().astype(np.float32)
    h /= (h.sum() + 1e-6)
    return h

def get_features(img):
    """
    hena bngm3 kol el features f vector wa7ed
    el vector da elly bydkhol lel model
    """
    return np.concatenate([
        hog_multi_scale(img),
        lbp_features(img),
        rgb_hist(img),
        hsv_hist(img)
    ])

# ------------------------------------
# Camera part (Realtime classification)
# ------------------------------------

# bnft7 el camera (0 = default webcam)
cam = cv2.VideoCapture(0)
print("[INFO] Camera started - press q to exit")

# loop sh8al tol ma el camera sh8ala
while True:

    # bn2ra frame frame mn el camera
    ret, frame = cam.read()

    # law el frame ma et2rash sah
    if not ret:
        break

    # bnbda2 n7sb el time 3shan FPS
    start = time.time()

    # OpenCV by3ml capture BGR fa bn7wlo RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # bnghz el sora zay el training
    fixed_img = prepare_image(rgb_frame)

    # bn3ml feature extraction
    feat = get_features(fixed_img)

    # reshape 3shan el model y2raha
    feat = feat.reshape(1, -1)

    # bn-apply scaler nafso elly et3lm
    feat = scale_tool.transform(feat)

    # bn-apply PCA nafso
    feat = pca_tool.transform(feat)

    # model bytl3 probabilities
    probs = svm_model.predict_proba(feat)[0]

    # a3la confidence
    best_prob = np.max(probs)

    # law el confidence wate
    if best_prob < unknown_limit:
        final_label = "Unknown"
        conf = best_prob
    else:
        idx = np.argmax(probs)
        final_label = labels_names[idx]
        conf = probs[idx]

    # bn7sb el FPS
    end = time.time()
    fps = 1 / (end - start + 1e-6)

    # bnktb el class name 3la el screen
    cv2.putText(
        frame,
        f"{final_label} ({conf:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # bnktb el FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # bn3rd el video
    cv2.imshow("Material Stream Identification", frame)

    # law dost q bn2fl el program
    key = cv2.waitKey(10) & 0xFF   # 10ms instead of 1
    if key == ord('q'):
        break

# ------------------------------------
# Close camera and windows
# ------------------------------------
cam.release()          # bn2fl el camera
cv2.destroyAllWindows()  # bn2fl ay window fat7a
