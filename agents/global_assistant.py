# agents/global_assistant.py
# Assistant global agentique : route image / question texte

from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from rag.rag_agent import AgentRAG
from agents.langgraph_orchestrator import app as diag_app  # ton graph image complet


# ----------------------------
# 1) Définition de l'état global
# ----------------------------

class AssistantState(TypedDict, total=False):
    user_message: str                 # question utilisateur (texte)
    image_path: Optional[str]         # chemin d'image si présent
    mode: str                         # "image" ou "rag"
    diag_report: Optional[str]        # rapport complet du pipeline image
    rag_answer: Optional[str]         # réponse du RAG texte
    final_answer: Optional[str]       # réponse finale pour le frontend


# ----------------------------
# 2) Initialisation Agent RAG
# ----------------------------

agent_rag = AgentRAG()


# ----------------------------
# 3) Nœuds du graph
# ----------------------------

def router_node(state: AssistantState) -> AssistantState:
    """
    Ne fait que passer l'état, la vraie décision
    est dans la fonction de routage `route_decision`.
    """
    return state


def route_decision(state: AssistantState) -> str:
    """
    Décide quel chemin prendre :
    - si une image est fournie -> "image"
    - sinon -> "rag"
    """
    if state.get("image_path"):
        return "image"
    return "rag"


def image_diagnosis_node(state: AssistantState) -> AssistantState:
    """
    Appelle ton graph image (DL + ML + Neo4j + RAG)
    déjà défini dans langgraph_orchestrator.
    """
    image_path = state.get("image_path")
    if not image_path:
        raise ValueError("Aucune image fournie pour le diagnostic.")

    # On invoque ton app image existante
    diag_state = diag_app.invoke({"image_path": image_path})

    # On récupère le rapport final
    state["diag_report"] = diag_state["final_report"]
    return state


def rag_qna_node(state: AssistantState) -> AssistantState:
    """
    Appelle l'AgentRAG pour répondre à une question textuelle
    sur la santé numérique.
    """
    question = state.get("user_message", "")
    if not question:
        raise ValueError("Aucune question fournie pour le RAG.")

    answer = agent_rag.ask(question)
    state["rag_answer"] = answer
    return state


def final_answer_node(state: AssistantState) -> AssistantState:
    """
    Construit une réponse finale propre pour le frontend.
    """
    if state.get("diag_report"):
        # Mode image : on renvoie directement le rapport complet
        state["final_answer"] = state["diag_report"]
    elif state.get("rag_answer"):
        # Mode question texte : on renvoie juste la réponse du RAG
        state["final_answer"] = state["rag_answer"]
    else:
        state["final_answer"] = "⚠ Aucun résultat généré par les agents."

    return state


# ----------------------------
# 4) Construction du graph global
# ----------------------------

workflow = StateGraph(AssistantState)

# Déclaration des nœuds
workflow.add_node("router", router_node)
workflow.add_node("image_diagnosis", image_diagnosis_node)
workflow.add_node("rag_qna", rag_qna_node)
workflow.add_node("final", final_answer_node)

# Point d'entrée
workflow.set_entry_point("router")

# Routage conditionnel
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "image": "image_diagnosis",
        "rag": "rag_qna",
    },
)

# Fin de chaque branche
workflow.add_edge("image_diagnosis", "final")
workflow.add_edge("rag_qna", "final")
workflow.add_edge("final", END)

# On compile le graph
assistant_app = workflow.compile()


# ----------------------------
# 5) Fonction utilitaire
# ----------------------------

def run_assistant(
    user_message: str = "",
    image_path: Optional[str] = None,
) -> str:
    """
    Fonction unique pour le frontend (Streamlit, API, etc.)

    - Si image_path != None  -> pipeline image
    - Sinon                  -> question RAG texte
    """
    init_state: AssistantState = {
        "user_message": user_message,
        "image_path": image_path,
    }

    final_state = assistant_app.invoke(init_state)
    return final_state.get("final_answer", "⚠ Pas de réponse générée.")


if __name__ == "__main__":
    # Exemple 1 : diagnostic par image
    from pprint import pprint

    print("=== TEST IMAGE ===")
    answer_img = run_assistant(
        user_message="Analyse cette radio.",
        image_path="person1_virus_6.jpeg"
    )
    print(answer_img)

    # Exemple 2 : question texte
    print("\n=== TEST TEXTE ===")
    answer_txt = run_assistant(
        user_message="Quels sont les avantages de la santé numérique ?"
    )
    print(answer_txt)
