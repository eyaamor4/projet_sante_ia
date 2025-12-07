# ============================================================
#  🔵 INFÉRENCE ML (SIFT + BoVW + SVM)
#  Charge les modèles sauvegardés et prédit une image.
# ============================================================

import os
import cv2
import numpy as np
import joblib

# ------------------------------------------------------------
# Chargement des modèles sauvegardés
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models_ml")

kmeans = joblib.load(os.path.join(MODEL_DIR, "bovw_kmeans.joblib"))
classifier = joblib.load(os.path.join(MODEL_DIR, "ml_classifier.joblib"))
class_names = joblib.load(os.path.join(MODEL_DIR, "class_names.joblib"))

K = kmeans.n_clusters
sift = cv2.SIFT_create()

# ------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ------------------------------------------------------------

def extract_sift_descriptors(image_path: str):
    """Extrait les descripteurs SIFT d'une image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Impossible de lire l'image : {image_path}")
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def build_histogram(descriptors, kmeans_model, K):
    """Construit l'histogramme BoVW d'une image."""
    hist = np.zeros(K)
    if descriptors is not None and len(descriptors) > 0:
        cluster_result = kmeans_model.predict(descriptors)
        for idx in cluster_result:
            hist[idx] += 1
    # Normalisation L2
    hist = hist / (np.linalg.norm(hist) + 1e-10)
    return hist.reshape(1, -1)

# ------------------------------------------------------------
# FONCTION DE PRÉDICTION
# ------------------------------------------------------------

def predict_image_ml(image_path: str):
    """
    Retourne :
        - label (str)
        - proba (float)
    """
    desc = extract_sift_descriptors(image_path)
    hist = build_histogram(desc, kmeans, K)

    # Certaines méthodes ont predict_proba(), d'autres non
    if hasattr(classifier, "predict_proba"):
        proba_vec = classifier.predict_proba(hist)[0]
        class_idx = int(np.argmax(proba_vec))
        proba = float(proba_vec[class_idx])
    else:
        class_idx = int(classifier.predict(hist)[0])
        proba = 1.0

    label = class_names[class_idx]
    return label, proba


# ------------------------------------------------------------
# TEST LOCAL
# ------------------------------------------------------------
if __name__ == "__main__":
    test_img = "test.jpg"  # mets une vraie image ici
    label, proba = predict_image_ml(test_img)
    print(f"🔍 ML Prediction: {label}  |  confidence = {proba:.3f}")
