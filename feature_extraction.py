import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

# ------------------------------------------------------------------
# -----------------------------------------------------------------------
img_dim = (128, 128)       # el size da wa7ed ll images kolaha 3shan el features teb2a consistent

AUG_folder = "augmented_dataset" 
ORG_folder = "dataset"            


# class names 
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]
# tartib el classes mohem 3shan el labels teb2a mazbota

# -------------------------------------------------
# image preparation
# -------------------------------------------------
def fix_image(img):
    """
    Resize image and normalize it

    bna3mel resize 3shan kol el images teb2a nafs el size  
    bna2sm 3la 255 3shan el values teb2a ben 0 w 1  
    da by5aly el training ashl w asr3
    """
    img = cv2.resize(img, img_dim)
    img = img.astype(np.float32)
    img = img / 255.0
    return img

# -------------------------------------------------
# HOG feature extraction
# -------------------------------------------------
def hog_single(img):
    """
    hena bntl3 hog features mn sora wa7da  
    awl 7aga bn7wlha grayscale  
    ba3d kda hog bytl3 features el shape w el edges
    """
    # bnrg3 el image ll range 0-255 3shan OpenCV y3rf yshtaghal
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # hog function btst5dm gradients w directions
    f = hog(
        gray,
        orientations=9,          # 3dd el directions
        pixels_per_cell=(16, 16),# size el cell
        cells_per_block=(2, 2),  # grouping el cells
        block_norm="L2-Hys",
        feature_vector=True      # 3shan yrg3 vector wa7ed
    )
    return f

def hog_two_scales(img):
    """
    bnst5dm hog 3la scale kbir w scale asghar  
    el idea en el image momkn teb2a details ktera aw 2lela  
    fa bngrb aktar mn size
    """
    # hog mn el image el aslya
    f_big = hog_single(img)

    # bn3ml resize ll image size asghar
    small_img = cv2.resize(img, (64, 64))
    small_img = cv2.resize(small_img, img_dim)

    # hog mn el image el soghayara
    f_small = hog_single(small_img)

    # bngm3 el 2 vectors f vector wa7ed
    return np.concatenate((f_big, f_small))

# -------------------------------------------------
# LBP feature
# -------------------------------------------------
def lbp_histogram(img):
    """
    LBP bt3br 3n el texture  
    msl textures bta3t el paper aw cardboard  
    btfr2 ben el materials
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # local binary pattern
    lbp_img = local_binary_pattern(gray, 8, 2, method="uniform")

    # bn7sb histogram ll LBP values
    hist, _ = np.histogram(lbp_img, bins=10, range=(0, 10))
    hist = hist.astype(np.float32)

    # normalization 3shan el values mat2srsh
    hist = hist / (hist.sum() + 1e-6)
    return hist

# -------------------------------------------------
# RGB color histogram
# -------------------------------------------------
def rgb_color_hist(img):
    """
    hena bnshof el colors distribution bel RGB  
    msl el plastic lono byb2a wa7ed aktar
    """
    img = (img * 255).astype(np.uint8)

    h = cv2.calcHist(
        [img],
        [0, 1, 2],   # R G B channels
        None,
        (8, 8, 8),
        [0, 256, 0, 256, 0, 256]
    )

    h = h.flatten().astype(np.float32)
    h = h / (h.sum() + 1e-6)
    return h

# -------------------------------------------------
# HSV color histogram
# -------------------------------------------------
def hsv_color_hist(img):
    """
    HSV bykon ahsn f conditions el light el mo5tlfa  
    fa da bysa3d el model yfhm el image aktar
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
    h = h / (h.sum() + 1e-6)
    return h

# -------------------------------------------------
# combine all features together
# -------------------------------------------------
def build_feature_vector(img):
    """
    hena bngm3 kol el features  
    hog + lbp + rgb + hsv  
    3shan el model yakhod sora kamla mn kol el zwaya
    """
    features = []

    features.append(hog_two_scales(img))   # shape features
    features.append(lbp_histogram(img))    # texture features
    features.append(rgb_color_hist(img))   # color RGB
    features.append(hsv_color_hist(img))   # color HSV

    return np.concatenate(features)

# -------------------------------------------------
# main feature extraction loop
# -------------------------------------------------
X_list = []   # hena hn5zn el feature vectors
y_list = []   # hena hn5zn el labels

print("\nStarting feature extraction...\n")

# bnlf 3la el augmented w el original dataset
for current_folder in [AUG_folder, ORG_folder]:

    # bnlf 3la kol class
    for class_index in range(len(CLASSES)):
        class_name = CLASSES[class_index]
        class_path = os.path.join(current_folder, class_name)

        all_images = os.listdir(class_path)
        print(f"Class {class_name}: {len(all_images)} images")

        # bnlf 3la kol sora gwa el class
        for img_name in tqdm(all_images):

            # bn-skip ay file msh image
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            full_path = os.path.join(class_path, img_name)
            img = cv2.imread(full_path)

            # lw el sora byza msh ma2roya
            if img is None:
                continue

            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocessing
            img = fix_image(img)

            # feature extraction
            vec = build_feature_vector(img)

            # store data
            X_list.append(vec)
            y_list.append(class_index)

# -------------------------------------------------
# convert to numpy arrays
# -------------------------------------------------
X_list = np.array(X_list, dtype=np.float32)
y_list = np.array(y_list, dtype=np.int64)

print("\nExtraction finished")
print("X shape:", X_list.shape)
print("y shape:", y_list.shape)

# -------------------------------------------------
# save features
# -------------------------------------------------
np.save("X_features.npy", X_list)
np.save("y_labels.npy", y_list)

print("\nFiles saved successfully")
