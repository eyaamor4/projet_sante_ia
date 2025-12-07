import cv2
import numpy as np
import joblib
import logging
import os
logging.basicConfig(level=logging.INFO)

class AgentML:
    def __init__(self):
        logging.info("🔧 Initialisation Agent ML...")
        

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ML_DIR = os.path.join(BASE_DIR, "models", "saved_models_ml")

        vocab_path = os.path.join(ML_DIR, "bovw_kmeans.joblib")
        clf_path = os.path.join(ML_DIR, "ml_classifier.joblib")
        names_path = os.path.join(ML_DIR, "class_names.joblib")


        # Charger modèles ML
        self.kmeans = joblib.load(vocab_path)
        self.classifier = joblib.load(clf_path)
        self.class_names = joblib.load(names_path)

        # SIFT
        self.sift = cv2.SIFT_create()

        logging.info("✓ Agent ML initialisé avec succès!")

    # --------------------------------------------------
    # 1) Extraire les descripteurs SIFT d'une image
    # --------------------------------------------------
    def extract_sift_features(self, img):
        """
        Retourne un histogramme BoVW représentant l'image.
        """
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        if descriptors is None:
            # Si SIFT ne trouve rien
            descriptors = np.zeros((1, 128), dtype=np.float32)

        # Convertir en histogramme BoVW
        labels = self.kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=self.kmeans.n_clusters)

        hist = hist / np.linalg.norm(hist) if np.linalg.norm(hist) != 0 else hist
        return hist

    # --------------------------------------------------
    # 2) Prédire avec SVM + BoVW
    # --------------------------------------------------
    def predict_image_ml(self, img):
        """
        Analyse ML BoVW + SVM et retourne un dict standardisé
        compatible LangGraph + Orchestrateur.
        """

        # Extraire histogramme
        features = self.extract_sift_features(img)

        # Prédiction SVM
        probs = self.classifier.predict_proba([features])[0]

        normal_prob = float(probs[0])
        pneumonia_prob = float(probs[1])

        prediction = "PNEUMONIA" if pneumonia_prob > normal_prob else "NORMAL"
        confidence = max(normal_prob, pneumonia_prob)

        logging.info(f"✓ ML → {prediction} ({confidence*100:.2f}%)")

        return {
            "agent": "ml",
            "diagnosis": prediction,
            "confidence": confidence,
            "probabilities": {
                "NORMAL": normal_prob,
                "PNEUMONIA": pneumonia_prob
            }
        }
