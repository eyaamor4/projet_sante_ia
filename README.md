# 🧠 Assistant IA Santé Numérique — Multi-Agents (RAG + Vision + Streamlit)

Ce projet implémente un **assistant intelligent pour l’aide au diagnostic médical**, basé sur :
- 🧩 **Architecture multi-agents**  
- 🔍 **RAG (Retrieval Augmented Generation)**  
- 👁️ **Deep Learning Vision **  
- 🗂️ **Base de données RAG avec pgvector**  
- 🌐 **Interface utilisateur Streamlit**

L’objectif est de combiner **IA Générative**, **vision médicale**, et **agents autonomes** pour fournir un système modulaire capable d’analyser des images radiologiques, extraire des connaissances, et répondre à des questions sur la santé numérique.

---

## 🚀 Fonctionnalités principales

### 🔬 1. **Agent de Diagnostic par Image (Rayon X Poumons)**
- Classification *NORMAL vs PNEUMONIA*
- Modèle MobileNetV2 optimisé
- Visualisation explicative avec **Grad-CAM**
- Pipeline complet de prétraitement d’image (OpenCV)

---

### 🧠 2. **Agent RAG Santé Numérique**
- Base de documents indexés avec **pgvector**
- Recherche sémantique (cosine similarity)
- Réponses enrichies via LLM (open-source uniquement)
- Agents orchestrés via LangGraph / LangChain

---

### 🤖 3. **Agent Global (Orchestrateur)**
Rôle :
- Router les requêtes
- Appeler les agents Vision ou RAG selon le besoin
- Fusionner les réponses dans un format structuré

---

### 💻 4. **Interface Web (Streamlit)**
- Upload d’image
- Affichage Grad-CAM
- Résultats du modèle
- Chat interface pour questions santé numérique

---



