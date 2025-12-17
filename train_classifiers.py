import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.decomposition import PCA


# --------------------------
# Parameters
# --------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
UNKNOWN_THRESHOLD = 0.4  # Confidence threshold for "Unknown" class (ID=6)

# --------------------------
# Load features and labels
# --------------------------
X = np.load("X_features.npy")  # [num_samples, num_features]
y = np.load("y_labels.npy")    # [num_samples]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# --------------------------
# Split into Train / Validation sets
# --------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# --------------------------
# Feature Scaling
# --------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)



pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

print("Reduced dims:", X_train_pca.shape[1])

# --------------------------
# Train SVM Classifier
# --------------------------
svm_clf = SVC(
    kernel="rbf",
    C=10,
    gamma='scale',
    class_weight="balanced",
    probability=True
)

svm_clf.fit(X_train_pca, y_train)


# --------------------------
# Train k-NN Classifier
# --------------------------
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance',metric='cosine')
knn_clf.fit(X_train_pca, y_train)

# --------------------------
# Unknown Class Handling
# --------------------------
def predict_with_rejection(model, X, threshold=UNKNOWN_THRESHOLD):
    """
    Predict class labels with rejection for low-confidence predictions.
    Returns 6 for unknown predictions.
    """
    probs = model.predict_proba(X)
    preds = []
    classes = model.classes_
    for p in probs:
        max_prob = max(p)
        if max_prob < threshold:
            preds.append(6)  # Unknown class
        else:
            preds.append(classes[np.argmax(p)])
    return np.array(preds)

# --------------------------
# SVM Predictions
# --------------------------
y_pred_svm = predict_with_rejection(svm_clf, X_val_pca, threshold=UNKNOWN_THRESHOLD)
print("\nSVM Classifier Results (with Unknown Handling):")
print("Accuracy:", accuracy_score(y_val, y_pred_svm))
print(classification_report(y_val, y_pred_svm, zero_division=0))

# --------------------------
# k-NN Predictions
# --------------------------
y_pred_knn = predict_with_rejection(knn_clf, X_val_pca, threshold=UNKNOWN_THRESHOLD)
print("\nk-NN Classifier Results (with Unknown Handling):")
print("Accuracy:", accuracy_score(y_val, y_pred_knn))
print(classification_report(y_val, y_pred_knn, zero_division=0))

# --------------------------
# Save models and scaler
# --------------------------
joblib.dump(svm_clf, "svm_model.pkl")
joblib.dump(knn_clf, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler for later use
joblib.dump(pca, "pca.pkl")

print("\nTrained models, scaler, and PCA saved successfully.")