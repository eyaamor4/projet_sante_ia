# -*- coding: utf-8 -*-
"""
Agent pgVector pour recherche par similarité
Utilise les embeddings de MobileNetV2
Architecture Multi-Agents - Projet Data Science GI3
VERSION OPTIMISÉE
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import psycopg2
from datetime import datetime
from agents.agent_dl import AgentDL


# Configuration de l'encodage
os.environ['PGCLIENTENCODING'] = 'UTF8'

# Ajouter le chemin racine
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from agents.agent_dl import AgentDL

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentPgVector:
    """Agent de recherche par similarité avec pgVector"""
    
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "medical_xray;",
        db_user: str = "postgres",
        db_password: str = "postgres",
        agent_id: str = "AgentPgVector_001"
    ):
        """Initialise l'Agent pgVector"""
        self.agent_id = agent_id
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password,
            "client_encoding": "utf8"
        }
        
        # Statistiques
        self.stats = {
            "total_searches": 0,
            "total_images_added": 0,
            "average_similarity": []
        }
        
        try:
            # Connexion PostgreSQL
            self.conn = psycopg2.connect(**self.db_config)
            logger.info(f"✓ {self.agent_id} connecté à PostgreSQL")
            
            # ✅ CORRECTION : Créer UNE SEULE instance d'AgentDL
            self.agent_dl = AgentDL()
            logger.info(f"✓ AgentDL chargé pour embeddings")
            
            # Initialiser le modèle d'embedding
            self._init_embedding_model()
            
            logger.info(f"✓ {self.agent_id} initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion PostgreSQL: {e}")
            raise
    
    
    def _init_embedding_model(self):
        """Initialise le modèle d'extraction d'embeddings"""
        try:
            from tensorflow.keras.models import Model
            
            # ✅ CORRECTION : Utiliser l'instance existante
            full_model = self.agent_dl.model
            
            # Créer un modèle jusqu'à l'avant-dernière couche
            # MobileNetV2 : avant-dernière couche = GlobalAveragePooling2D (1280D)
            self.embedding_model = Model(
                inputs=full_model.input,
                outputs=full_model.layers[-2].output
            )
            
            logger.info(f"  Modèle d'embedding: {self.embedding_model.layers[-1].name}")
            logger.info(f"  Dimension embedding: {self.embedding_model.output_shape}")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation modèle embedding: {e}")
            raise
    
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extrait l'embedding d'une image (vecteur 1280D)
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Vecteur embedding normalisé
        """
        try:
            # ✅ CORRECTION : Utiliser l'instance existante
            img_array = self.agent_dl.preprocess_image(image_path)
            
            # Extraction de features
            embedding = self.embedding_model.predict(img_array, verbose=0)
            embedding = embedding.flatten()
            
            # Normalisation L2 (pour distance cosinus)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction embedding: {e}")
            raise
    
    
    def add_image(
        self,
        image_path: str,
        diagnosis: str,
        patient_age: int = None,
        patient_gender: str = None,
        symptoms: str = None
    ) -> int:
        """
        Ajoute une image à la base de données
        
        Args:
            image_path: Chemin vers l'image
            diagnosis: Diagnostic ('PNEUMONIA' ou 'NORMAL')
            patient_age: Âge du patient
            patient_gender: Genre du patient ('M' ou 'F')
            symptoms: Symptômes du patient
            
        Returns:
            ID de l'image ajoutée (ou -1 si erreur)
        """
        try:
            # Vérifier que l'image existe
            if not os.path.exists(image_path):
                logger.error(f"Image non trouvée: {image_path}")
                return -1
            
            # Vérifier le diagnostic
            if diagnosis not in ['PNEUMONIA', 'NORMAL']:
                logger.error(f"Diagnostic invalide: {diagnosis}")
                return -1
            
            # Extraire l'embedding
            embedding = self.extract_embedding(image_path)
            
            # Insérer dans PostgreSQL
            cursor = self.conn.cursor()
            
            # Vérifier si l'image existe déjà
            cursor.execute(
                "SELECT id FROM medical_images WHERE image_path = %s",
                (image_path,)
            )
            existing = cursor.fetchone()
            
            if existing:
                logger.warning(f"Image déjà présente (ID={existing[0]})")
                cursor.close()
                return existing[0]
            
            # Insérer nouvelle image
            cursor.execute("""
                INSERT INTO medical_images 
                (image_path, diagnosis, embedding, patient_age, patient_gender, symptoms)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                image_path,
                diagnosis,
                embedding.tolist(),
                patient_age,
                patient_gender,
                symptoms
            ))
            
            image_id = cursor.fetchone()[0]
            self.conn.commit()
            cursor.close()
            
            self.stats["total_images_added"] += 1
            logger.info(f"✓ Image ajoutée: ID={image_id}, diagnosis={diagnosis}")
            
            return image_id
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout image: {e}")
            self.conn.rollback()
            return -1
    
    
    def find_similar(
        self,
        image_path: str,
        top_k: int = 5,
        diagnosis_filter: str = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Trouve les cas similaires dans la base
        
        Args:
            image_path: Chemin vers l'image requête
            top_k: Nombre de résultats à retourner
            diagnosis_filter: Filtrer par diagnostic ('PNEUMONIA' ou 'NORMAL')
            min_similarity: Similarité minimum (0-1)
            
        Returns:
            Liste des cas similaires
        """
        try:
            # Vérifier que la base contient des images
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM medical_images")
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.warning("⚠ Base de données vide")
                cursor.close()
                return []
            
            logger.info(f"Recherche de similarité (base: {count} images)...")
            
            # Extraire l'embedding de l'image requête
            query_embedding = self.extract_embedding(image_path)
            
            # Requête SQL avec distance cosinus
            if diagnosis_filter:
                cursor.execute("""
                    SELECT 
                        id,
                        image_path,
                        diagnosis,
                        patient_age,
                        patient_gender,
                        symptoms,
                        1 - (embedding <=> %s::vector) / 2 AS similarity,
                        created_at
                    FROM medical_images
                    WHERE diagnosis = %s AND (1 - (embedding <=> %s::vector) / 2) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (
                    query_embedding.tolist(),
                    diagnosis_filter,
                    query_embedding.tolist(),
                    min_similarity,
                    query_embedding.tolist(),
                    top_k
                ))
            else:
                cursor.execute("""
                    SELECT 
                        id,
                        image_path,
                        diagnosis,
                        patient_age,
                        patient_gender,
                        symptoms,
                        1 - (embedding <=> %s::vector) / 2 AS similarity,
                        created_at
                    FROM medical_images
                    WHERE (1 - (embedding <=> %s::vector) / 2) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (
                    query_embedding.tolist(),
                    query_embedding.tolist(),
                    min_similarity,
                    query_embedding.tolist(),
                    top_k
                ))
            
            results = []
            for row in cursor.fetchall():
                similarity = float(row[6])
                results.append({
                    "id": row[0],
                    "image_path": row[1],
                    "diagnosis": row[2],
                    "patient_age": row[3],
                    "patient_gender": row[4],
                    "symptoms": row[5],
                    "similarity": round(similarity, 4),
                    "created_at": row[7].isoformat() if row[7] else None
                })
                
                self.stats["average_similarity"].append(similarity)
            
            cursor.close()
            self.stats["total_searches"] += 1
            
            logger.info(f"✓ Trouvé {len(results)} cas similaires")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche similarité: {e}")
            return []
    
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la base de données"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM medical_images")
            total = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT diagnosis, COUNT(*) 
                FROM medical_images 
                GROUP BY diagnosis
            """)
            by_diagnosis = dict(cursor.fetchall())
            
            cursor.execute("SELECT MAX(created_at) FROM medical_images")
            last_insert = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                "total_images": total,
                "pneumonia_cases": by_diagnosis.get('PNEUMONIA', 0),
                "normal_cases": by_diagnosis.get('NORMAL', 0),
                "last_insert": last_insert.isoformat() if last_insert else None
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur statistiques: {e}")
            return {}
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        stats = self.stats.copy()
        
        if stats["average_similarity"]:
            stats["avg_similarity"] = round(np.mean(stats["average_similarity"]), 4)
            stats["min_similarity"] = round(np.min(stats["average_similarity"]), 4)
            stats["max_similarity"] = round(np.max(stats["average_similarity"]), 4)
        else:
            stats["avg_similarity"] = 0.0
            stats["min_similarity"] = 0.0
            stats["max_similarity"] = 0.0
        
        del stats["average_similarity"]
        stats["database"] = self.get_database_stats()
        
        return stats
    
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Connexion PostgreSQL fermée")


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_agent_pgvector():
    """Test de l'Agent pgVector"""
    
    print("\n" + "="*80)
    print("🧪 TEST DE L'AGENT PGVECTOR")
    print("="*80 + "\n")
    
    try:
        print("--- Initialisation ---")
        agent = AgentPgVector(db_password="postgres")
        
        print("\n--- Statistiques Base de Données ---")
        db_stats = agent.get_database_stats()
        print(f"  Total images: {db_stats['total_images']}")
        print(f"  Cas PNEUMONIA: {db_stats['pneumonia_cases']}")
        print(f"  Cas NORMAL: {db_stats['normal_cases']}")
        
        test_image = "testH.jpeg"
        
        if os.path.exists(test_image):
            print(f"\n--- Test Ajout Image ---")
            image_id = agent.add_image(
                image_path=test_image,
                diagnosis="PNEUMONIA",
                patient_age=65,
                symptoms="Toux, fièvre"
            )
            print(f"  Image ID: {image_id}")
            
            print(f"\n--- Test Recherche Similarité ---")
            similar_cases = agent.find_similar(image_path=test_image, top_k=3)
            
            print(f"\n  Résultats ({len(similar_cases)} cas similaires):")
            for i, case in enumerate(similar_cases, 1):
                print(f"\n  {i}. ID={case['id']} - {case['diagnosis']}")
                print(f"     Similarité: {case['similarity']:.2%}")
                print(f"     Image: {os.path.basename(case['image_path'])}")
                if case['patient_age']:
                    print(f"     Âge: {case['patient_age']} ans")
        else:
            print(f"\n⚠️  Image de test non trouvée: {test_image}")
        
        print("\n--- Statistiques Agent ---")
        stats = agent.get_stats()
        print(f"  Recherches effectuées: {stats['total_searches']}")
        print(f"  Images ajoutées: {stats['total_images_added']}")
        if stats['avg_similarity'] > 0:
            print(f"  Similarité moyenne: {stats['avg_similarity']:.2%}")
        
        agent.close()
        
        print("\n" + "="*80)
        print("✅ TEST TERMINÉ AVEC SUCCÈS")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent_pgvector()