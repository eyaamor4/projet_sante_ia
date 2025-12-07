# -*- coding: utf-8 -*-
"""
Agent Deep Learning (Agent DL)
Utilise MobileNetV2 pour la classification d'images Chest X-Ray
Architecture Multi-Agents - Projet Data Science GI3
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from datetime import datetime
import tempfile
from PIL import Image

# TensorFlow et Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Gestion des données
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("INFO: Le module 'joblib' n'est pas installé. Les métadonnées seront ignorées.")

# XAI (Explainabilité) - Optionnel
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import matplotlib.pyplot as plt
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("INFO: Les modules 'lime' ou 'matplotlib' ne sont pas installés. XAI désactivé.")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentDL:
    """Agent Deep Learning pour classification d'images médicales."""
    
    def __init__(
        self,
        model_path: str = 'models/mobilenet_phase2_best (1).keras',
        agent_id: str = 'AgentDL_001'
    ):
        """Initialise l'Agent Deep Learning."""
        self.agent_id = agent_id
        self.model_path = model_path
        self.model = None
        self.class_names = ['NORMAL', 'PNEUMONIA']  # Classes par défaut
        self.metadata = None
        self.img_size = (224, 224)
        self.last_conv_layer_name = None
        
        # Statistiques
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_inference_time': 0.0,
            'confidence_scores': []
        }
        
        self._initialize()
        logger.info(f"✓ {self.agent_id} initialisé avec succès")
    
    def _initialize(self):
        """Initialise le modèle et les métadonnées."""
        try:
            self._load_model()
            self._find_last_conv_layer()
            self._check_device()
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            raise
    
    def _load_model(self):
        """Charge le modèle Deep Learning."""
        path = self.model_path
        
        if not os.path.exists(path):
            # Rechercher dans les répertoires parents
            possible_paths = [path, os.path.join('../', path), os.path.join('../../', path)]
            found_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
            if found_path is None:
                raise FileNotFoundError(
                    f"Modèle non trouvé: {self.model_path}\n"
                    f"Vérifiez que le fichier existe dans le dossier 'models/'"
                )
            self.model_path = found_path
            
        logger.info(f"Chargement du modèle: {self.model_path}")
        self.model = load_model(self.model_path)
        logger.info(f"✓ Modèle chargé: {self.model.name}")
        logger.info(f"  Input shape: {self.model.input_shape}")
    
    def _find_last_conv_layer(self):
        """Détecte la dernière couche convolutive pour Grad-CAM."""
        conv_layer = None
        for layer in reversed(self.model.layers):
            try:
                # Vérifier si c'est une couche convolutive
                if ('conv' in layer.name.lower() or 'features' in layer.name.lower()) and 'bn' not in layer.name.lower():
                    # Utiliser layer.output.shape au lieu de layer.output_shape
                    output_shape = layer.output.shape if hasattr(layer, 'output') else None
                    if output_shape and len(output_shape) == 4:
                        conv_layer = layer.name
                        break
            except Exception as e:
                continue
                    
        self.last_conv_layer_name = conv_layer
        logger.info(f"  Dernière couche Conv (Grad-CAM): {self.last_conv_layer_name or 'Non trouvée'}")

    def _check_device(self):
        """Vérifie la disponibilité du GPU."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"✓ GPU détecté: {len(gpus)} device(s)")
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                logger.warning(f"⚠ Erreur de configuration GPU: {e}")
        else:
            logger.info("ℹ Aucun GPU détecté - Utilisation du CPU")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Prétraite une image pour la prédiction."""
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement image {image_path}: {e}")
            raise
    
    def predict(
        self,
        image_path: str,
        return_probabilities: bool = False,
        explain: bool = False
    ) -> Dict[str, Any]:
        """Effectue une prédiction sur une image."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image non trouvée: {image_path}")
            
            img_array = self.preprocess_image(image_path)
            
            prediction_start = datetime.now()
            prediction = self.model.predict(img_array, verbose=0)
            inference_time = (datetime.now() - prediction_start).total_seconds() * 1000
            
            # Interpréter la prédiction binaire
            prob = float(prediction[0][0])
            predicted_class_idx = 1 if prob > 0.5 else 0
            
            predicted_class = self.class_names[predicted_class_idx]
            confidence = prob if predicted_class_idx == 1 else (1 - prob)
            
            self._update_stats(inference_time, confidence)
            
            result = {
                'agent_id': self.agent_id,
                'agent_type': 'DeepLearning',
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'raw_probability': prob,
                'inference_time_ms': round(inference_time, 2),
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model.name if hasattr(self.model, 'name') else 'MobileNetV2'
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    self.class_names[0]: float(1 - prob),
                    self.class_names[1]: float(prob)
                }
            
            if explain:
                result['explanation_requested'] = True
            
            logger.info(
                f"✓ Prédiction: {predicted_class} "
                f"(confiance: {confidence:.2%}, temps: {inference_time:.2f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction {image_path}: {e}")
            self.stats['failed_predictions'] += 1
            
            return {
                'agent_id': self.agent_id,
                'agent_type': 'DeepLearning',
                'error': str(e),
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(
        self,
        image_paths: List[str],
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """Effectue des prédictions sur un lot d'images."""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, return_probabilities=return_probabilities)
            results.append(result)
        return results

    def explain_prediction(
        self,
        image_path: str,
        num_samples: int = 500
    ) -> Optional[Dict[str, Any]]:
        """Génère une explication LIME pour la prédiction."""
        if not LIME_AVAILABLE:
            logger.warning("⚠ LIME non disponible")
            return None
        
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img).astype(np.double)
            
            def predict_fn(images):
                images = images / 255.0
                preds = self.model.predict(images, verbose=0)
                prob_pneumonia = preds[:, 0].reshape(-1, 1)
                prob_normal = 1 - prob_pneumonia
                return np.hstack([prob_normal, prob_pneumonia])
            
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                predict_fn,
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=num_samples
            )
            
            predicted_index = np.argmax(predict_fn(np.expand_dims(img_array, axis=0)))
            
            temp, mask = explanation.get_image_and_mask(
                predicted_index,
                positive_only=False,
                num_features=10,
                hide_rest=False
            )
            
            lime_image_path = self._save_lime_image(temp, mask, self.class_names[predicted_index])
            
            logger.info(f"✓ Explication LIME générée: {lime_image_path}")
            
            return {
                'method': 'LIME',
                'num_samples': num_samples,
                'predicted_class_index': predicted_index,
                'explanation_image_path': lime_image_path
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération LIME: {e}")
            return None
            
    def _save_lime_image(self, temp_img: np.ndarray, mask: np.ndarray, predicted_class: str) -> str:
        """Sauvegarde l'image LIME."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(mark_boundaries(temp_img / 2 + 0.5, mask))
            ax.set_title(f"LIME: {predicted_class}", fontsize=12)
            ax.axis('off')
            
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"lime_{predicted_class}_{datetime.now().strftime('%H%M%S')}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            


            return output_path
        except Exception as e:
            logger.error(f"Erreur sauvegarde LIME: {e}")
            return "Error saving LIME image"

    def generate_gradcam(self, image_path: str) -> Optional[np.ndarray]:
        """Génère une heatmap Grad-CAM."""
        if self.last_conv_layer_name is None:
            logger.warning("⚠ Grad-CAM impossible: couche conv. non trouvée")
            return None
        
        try:
            img_array = self.preprocess_image(image_path)
            
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, model_predictions = grad_model(img_array)
                loss = model_predictions[:, 0]
                
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            logger.info("✓ Heatmap Grad-CAM générée")
            return heatmap.numpy()
            
        except Exception as e:
            logger.error(f"❌ Erreur Grad-CAM: {e}")
            return None
    
    def overlay_heatmap(self, image_path: str, heatmap: np.ndarray, alpha: float = 0.4) -> Optional[str]:
        """Superpose la heatmap Grad-CAM à l'image originale."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Impossible de charger l'image: {image_path}")
                return None
            
            heatmap = cv2.resize(heatmap, self.img_size)
            resized_img = cv2.resize(img, self.img_size)
            
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            superimposed_img = cv2.addWeighted(resized_img, 1 - alpha, heatmap, alpha, 0)
            
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path).replace('.', '_')
            output_path = os.path.join(output_dir, f"gradcam_{base_name}.png")
            cv2.imwrite(output_path, superimposed_img)
            
            logger.info(f"✓ Grad-CAM sauvegardé: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Erreur overlay Grad-CAM: {e}")
            return None

    def _update_stats(self, inference_time: float, confidence: float):
        """Met à jour les statistiques."""
        self.stats['total_predictions'] += 1
        self.stats['successful_predictions'] += 1
        self.stats['confidence_scores'].append(confidence)
        
        n = self.stats['successful_predictions']
        avg = self.stats['average_inference_time']
        self.stats['average_inference_time'] = (avg * (n - 1) + inference_time) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        stats = self.stats.copy()
        if stats['confidence_scores']:
            stats['average_confidence'] = np.mean(stats['confidence_scores'])
            stats['min_confidence'] = np.min(stats['confidence_scores'])
            stats['max_confidence'] = np.max(stats['confidence_scores'])
            stats['std_confidence'] = np.std(stats['confidence_scores'])
        else:
            stats['average_confidence'] = 0.0
            stats['min_confidence'] = 0.0
            stats['max_confidence'] = 0.0
            stats['std_confidence'] = 0.0

        if stats['total_predictions'] > 0:
            stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions']
        else:
            stats['success_rate'] = 0.0
            
        del stats['confidence_scores']
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'agent."""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'DeepLearning',
            'model_name': self.model.name if hasattr(self.model, 'name') else 'Unknown',
            'model_path': self.model_path,
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params() if hasattr(self.model, 'count_params') else None,
            'img_size': self.img_size,
            'lime_available': LIME_AVAILABLE
        }


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_agent_dl():
    """Fonction de test pour l'Agent DL."""
    
    print("\n" + "=" * 80)
    print("TEST DE L'AGENT DEEP LEARNING (IMAGE UNIQUE)")
    print("=" * 80 + "\n")
    
    # CORRIGÉ: Nom exact du fichier avec l'espace
    MODEL_FILE = 'models/mobilenet_phase2_best (1).keras'
    
    # Image de test
    REAL_IMAGE_PATH = 'testH.jpeg'
    
    # Vérifier le modèle
    if not os.path.exists(MODEL_FILE):
        print(f"\n❌ ERREUR: Modèle non trouvé à '{MODEL_FILE}'.")
        print("Veuillez vérifier le nom et l'emplacement du fichier.")
        return
        
    # Vérifier l'image
    if not os.path.exists(REAL_IMAGE_PATH):
        print(f"\n⚠ AVERTISSEMENT: Image non trouvée à '{REAL_IMAGE_PATH}'.")
        print("Création d'une image factice temporaire...")
        try:
            dummy_image = np.random.rand(224, 224, 3) * 255
            dummy_image = dummy_image.astype(np.uint8)
            temp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(temp_dir, f"test_xray_{os.getpid()}.jpg")
            Image.fromarray(dummy_image).save(tmp_path)
            REAL_IMAGE_PATH = tmp_path
            print(f"  Utilisation de l'image temporaire: {tmp_path}")
        except Exception as e:
            print(f"  Erreur lors de la création de l'image factice: {e}")
            return

    tmp_path_to_delete = REAL_IMAGE_PATH if 'tempfile' in REAL_IMAGE_PATH else None

    try:
        # Initialiser l'agent
        print("\n--- Initialisation de l'Agent ---")
        agent = AgentDL(model_path=MODEL_FILE)
        
        # Informations Agent
        print("\n--- Informations Agent ---")
        info = agent.get_info()
        print(f"  Agent ID: {info.get('agent_id')}")
        print(f"  Modèle: {info.get('model_path')}")
        print(f"  Classes: {info.get('class_names')}")
        print(f"  Paramètres: {info.get('total_params'):,}")
        print(f"  Dernière couche Conv: {agent.last_conv_layer_name}")
        
        # Test Prédiction
        print(f"\n--- Test Prédiction sur {os.path.basename(REAL_IMAGE_PATH)} ---")
        result = agent.predict(REAL_IMAGE_PATH, return_probabilities=True)
        
        print("\n📊 Résultat de la Prédiction:")
        print(f"  Classe Prédite: {result.get('predicted_class')}")
        print(f"  Confiance: {result.get('confidence'):.4f} ({result.get('confidence')*100:.2f}%)")
        print(f"  Temps d'inférence: {result.get('inference_time_ms'):.2f} ms")
        
        if 'probabilities' in result:
            print("\n  Probabilités détaillées:")
            for cls, prob in result['probabilities'].items():
                bar = '█' * int(prob * 50)
                print(f"    {cls:10s}: {prob:.4f} |{bar}")
        
        # Test Grad-CAM
        print(f"\n--- Test Grad-CAM ---")
        heatmap = agent.generate_gradcam(REAL_IMAGE_PATH)
        if heatmap is not None:
            print(f"  ✓ Heatmap générée (shape: {heatmap.shape})")
            overlay_path = agent.overlay_heatmap(REAL_IMAGE_PATH, heatmap)
            if overlay_path:
                print(f"  ✓ Image Grad-CAM: {overlay_path}")
        else:
            print("  ⚠ Grad-CAM non disponible")

        # Test LIME (si disponible)
        if LIME_AVAILABLE:
            print(f"\n--- Test LIME ---")
            explanation = agent.explain_prediction(REAL_IMAGE_PATH, num_samples=100)
            if explanation:
                print(f"  ✓ Explication générée: {explanation.get('explanation_image_path')}")
        
        # Statistiques
        print("\n--- Statistiques ---")
        stats = agent.get_stats()
        print(f"  Total Prédictions: {stats.get('total_predictions')}")
        print(f"  Taux de succès: {stats.get('success_rate'):.2%}")
        print(f"  Temps moyen: {stats.get('average_inference_time'):.2f} ms")
        print(f"  Confiance moyenne: {stats.get('average_confidence'):.4f}")

        # Nettoyage
        if tmp_path_to_delete and os.path.exists(tmp_path_to_delete):
            os.unlink(tmp_path_to_delete)

        print("\n" + "=" * 80)
        print("✅ TEST TERMINÉ AVEC SUCCÈS!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent_dl()