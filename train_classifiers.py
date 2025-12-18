import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# ---------------------------------------------------------------------
# basic settings 
# -------------------------------------------------
test_ratio = 0.2
seed_value = 42
unknown_limit = 0.4   # law el confidence a2al mn kda n3tbr el image Unknown

# -------------------------------------------------------------
# load extracted features
# -------------------------------------------------
# X: features matrix
# y: labels
X_data = np.load("X_features.npy")
y_data = np.load("y_labels.npy")

print("Features shape:", X_data.shape)
print("Labels shape:", y_data.shape)

# -------------------------------------------------
# split data into training and validation
# --------------------------------------------------------
# bn2sm el data 80% train w 20% validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_data,
    y_data,
    test_size=test_ratio,
    random_state=seed_value,
    stratify=y_data   # 3shan el classes tfdl mtwazna
)

print("Train shape:", X_tr.shape)
print("Validation shape:", X_val.shape)

# -------------------------------------------------
# feature scaling
# --------------------------------------------------------------
# scaling mohem 3shan SVM w KNN ysht8lo  kwys
scaler_obj = StandardScaler()

X_tr = scaler_obj.fit_transform(X_tr)   # fit + transform on train
X_val = scaler_obj.transform(X_val)     # transform only on validation

# -------------------------------------------------
# PCA for dimensionality reduction
# ---------------------------------------------------------
# bn2ll 3dd el features bs mn8er ma ndy3 information
pca_obj = PCA(n_components=0.95, random_state=seed_value)

X_tr_pca = pca_obj.fit_transform(X_tr)
X_val_pca = pca_obj.transform(X_val)

print("Number of features after PCA:", X_tr_pca.shape[1])

# ------------------------------------------------------------
# SVM model training
# --------------------------------------------------
# SVM with RBF kernel
svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",  # 3shan yt3aml m3 el imbalance
    probability=True          # 3shan n3rf el confidence
)

svm_model.fit(X_tr_pca, y_tr)

# -------------------------------------------------
# KNN model training
# ---------------------------------------------------------------------------
# KNN based on cosine distance
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="cosine"
)

knn_model.fit(X_tr_pca, y_tr)

# -----------------------------------------------------------
# prediction with unknown handling
# -------------------------------------------------
def predict_with_unknown(model, features, limit=unknown_limit):
    """
    
    Predict class or return Unknown if confidence is low

    law el model msh wask f el prediction
    bnrg3 class Unknown (ID = 6)
    """
    all_probs = model.predict_proba(features)
    final_preds = []

    class_ids = model.classes_

    for prob in all_probs:
        highest = np.max(prob)

        if highest < limit:
            final_preds.append(6)  # Unknown
        else:
            idx = np.argmax(prob)
            final_preds.append(class_ids[idx])

    return np.array(final_preds)

# -------------------------------------------------
# evaluate SVM
# -----------------------------------------------------------------------
svm_preds = predict_with_unknown(svm_model, X_val_pca)

print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_val, svm_preds))
print(classification_report(y_val, svm_preds, zero_division=0))

# ----------------------------------------------------------------------------------------
# evaluate KNN
# -------------------------------------------------
knn_preds = predict_with_unknown(knn_model, X_val_pca)

print("\nKNN Results:")
print("Accuracy:", accuracy_score(y_val, knn_preds))
print(classification_report(y_val, knn_preds, zero_division=0))

# --------------------------------------------------------
# save models and preprocessing tools
# ----------------------------------------------------------------------------
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(scaler_obj, "scaler.pkl")
joblib.dump(pca_obj, "pca.pkl")

print("\nModels and preprocessing objects saved successfully")
