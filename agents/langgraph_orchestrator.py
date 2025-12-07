# -*- coding: utf-8 -*-
"""
LangGraph Orchestrator
Transforme ton orchestrateur Python en graph agentique
multi-agents : DL → ML → Fusion → PgVector → Graph → RAG
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
import logging
import cv2

# Import des agents
from agents.agent_dl import AgentDL
from agents.agent_ml import AgentML
from agents.agent_graph import AgentGraph
from rag.rag_agent import AgentRAG
from agents.agent_pgvector import AgentPgVector

# Initialisation des agents globaux
agent_rag = AgentRAG()
agent_pg = AgentPgVector()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# 1️⃣ ÉTAT DU GRAPH
# ----------------------------

class DiagnosticState(TypedDict):
    image_path: Optional[str]
    dl_result: Optional[Dict[str, Any]]
    ml_result: Optional[Dict[str, Any]]
    fusion: Optional[Dict[str, Any]]
    pgvector_results: Optional[Any]
    graph_info: Optional[Dict[str, Any]]
    rag_answer: Optional[str]
    final_report: Optional[str]


# ----------------------------
# 2️⃣ AGENTS
# ----------------------------

agent_dl = AgentDL(model_path="models/mobilenet_phase2_best (1).keras")
agent_ml = AgentML()
agent_graph = AgentGraph()


# ----------------------------
# 3️⃣ NODES DU GRAPH
# ----------------------------

# --- DL NODE ---
def run_agent_dl(state: DiagnosticState) -> DiagnosticState:
    img = state["image_path"]
    logger.info(f"[DL] Analyse de l'image {img}")

    state["dl_result"] = agent_dl.predict(img, return_probabilities=True)
    return state


# --- ML NODE ---
def run_agent_ml(state: DiagnosticState) -> DiagnosticState:
    img_path = state["image_path"]
    logger.info(f"[ML] Analyse ML de l'image {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Impossible de charger l'image : {img_path}")

    state["ml_result"] = agent_ml.predict_image_ml(img)
    return state


# --- FUSION NODE ---
def run_fusion(state: DiagnosticState) -> DiagnosticState:
    dl = state["dl_result"]
    ml = state["ml_result"]

    logger.info("[Fusion] Calcul du diagnostic final...")

    DL_WEIGHT = 0.6
    ML_WEIGHT = 0.4

    p_dl = dl["probabilities"]["PNEUMONIA"]
    p_ml = ml["probabilities"]["PNEUMONIA"]

    fusion_score = (DL_WEIGHT * p_dl) + (ML_WEIGHT * p_ml)
    final_class = "PNEUMONIA" if fusion_score > 0.5 else "NORMAL"

    state["fusion"] = {
        "final_class": final_class,
        "fusion_score": fusion_score
    }
    return state


# --- PGVECTOR NODE ---
def run_agent_pgvector(state: DiagnosticState) -> DiagnosticState:
    image_path = state["image_path"]
    diagnosis = state["fusion"]["final_class"]

    print(f"[PgVector] Ajout + similarité pour {image_path}")

    try:
        # 1) Ajouter l'image analysée dans la base si pas présente
        agent_pg.add_image(
            image_path=image_path,
            diagnosis=diagnosis
        )

        # 2) Trouver images similaires existantes
        state["pgvector_results"] = agent_pg.find_similar(
            image_path=image_path,
            top_k=3
        )

    except Exception as e:
        print("Erreur PgVector:", e)
        state["pgvector_results"] = []

    return state



# --- GRAPH NODE (NEO4J) ---
def run_agent_graph(state: DiagnosticState) -> DiagnosticState:
    diagnosis = state["fusion"]["final_class"]
    logger.info(f"[Neo4j] Récupération des connaissances pour {diagnosis}")

    state["graph_info"] = agent_graph.get_info(diagnosis)
    return state


# --- RAG NODE ---
def run_agent_rag(state: DiagnosticState) -> DiagnosticState:
    diagnosis = state["fusion"]["final_class"]
    question = f"Explique brièvement la maladie suivante : {diagnosis}"

    print(f"[RAG] Question envoyée : {question}")

    state["rag_answer"] = agent_rag.ask(question)
    return state


# --- FINAL REPORT NODE ---
def build_final_report(state: DiagnosticState) -> DiagnosticState:
    dl = state["dl_result"]
    ml = state["ml_result"]
    fusion = state["fusion"]
    graph = state["graph_info"]
    rag_answer = state["rag_answer"]
    pg = state.get("pgvector_results", [])

    # Format PgVector
    similar_text = "\n## Cas similaires trouvés (pgVector)\n"
    if pg:
        for case in pg:
            similar_text += (
                f"- ID {case['id']} | {case['diagnosis']} | "
                f"Similarité : {case['similarity']*100:.1f}%\n"
            )
    else:
        similar_text += "- Aucun cas similaire trouvé.\n"

    report = f"""
# Diagnostic Final — Assistant Santé Numérique

## Résultat final : **{fusion['final_class']}**
Score fusionné : {fusion['fusion_score']:.2%}

---

## 🔬 Détection DL (MobileNet)
Classe : {dl['predicted_class']}
Confiance : {dl['confidence']:.2%}

## 🤖 Détection ML
Classe : {ml['diagnosis']}
Confiance : {ml['confidence']:.2%}

---

{similar_text}

---

## 🧠 Informations Médicales (Knowledge Graph)
{graph}

---

## 📘 Explication détaillée via RAG
{rag_answer}

---

## Conclusion
Diagnostic généré par un pipeline multi-agents (DL + ML + PgVector + KG + RAG).
"""

    state["final_report"] = report
    return state


# ----------------------------
# 4️⃣ CONSTRUCTION DU GRAPH
# ----------------------------

workflow = StateGraph(DiagnosticState)

workflow.add_node("agent_dl", run_agent_dl)
workflow.add_node("agent_ml", run_agent_ml)
workflow.add_node("fusion", run_fusion)
workflow.add_node("pgvector", run_agent_pgvector)
workflow.add_node("agent_graph", run_agent_graph)
workflow.add_node("agent_rag", run_agent_rag)
workflow.add_node("final_report", build_final_report)

workflow.set_entry_point("agent_dl")

# ORCHESTRATION COMPLÈTE
workflow.add_edge("agent_dl", "agent_ml")
workflow.add_edge("agent_ml", "fusion")
workflow.add_edge("fusion", "pgvector")
workflow.add_edge("pgvector", "agent_graph")
workflow.add_edge("agent_graph", "agent_rag")
workflow.add_edge("agent_rag", "final_report")
workflow.add_edge("final_report", END)

app = workflow.compile()


# ----------------------------
# 5️⃣ UTILISATION
# ----------------------------

def diagnose_image(image_path: str):
    state = app.invoke({"image_path": image_path})
    return state["final_report"]


if __name__ == "__main__":
    print(diagnose_image("person1_virus_6.jpeg"))
