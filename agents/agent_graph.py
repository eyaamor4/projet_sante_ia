from neo4j import GraphDatabase

class AgentGraph:
    def __init__(self, uri="bolt://127.0.0.1:7687", user="neo4j", password="neo4jneo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_info(self, maladie: str):
        query = """
        MATCH (m:Maladie {name: $maladie})

        OPTIONAL MATCH (m)-[:PRESENTE_SYMPTOME]->(s:Symptome)
        OPTIONAL MATCH (m)-[:PRESENTE_SIGNE]->(sg:Signe)
        OPTIONAL MATCH (m)-[:A_RISQUE]->(r:Risque)
        OPTIONAL MATCH (m)-[:TRAITE_AVEC]->(t:Traitement)

        RETURN 
            collect(DISTINCT s.name) AS symptomes,
            collect(DISTINCT sg.name) AS signes,
            collect(DISTINCT r.name) AS risques,
            collect(DISTINCT t.name) AS traitements
        """

        with self.driver.session(database="neo4j") as session:
            result = session.run(query, maladie=maladie).single()

        return {
            "agent": "Agent_Graph",
            "maladie": maladie,
            "symptomes": [s for s in result["symptomes"] if s],
            "signes": [s for s in result["signes"] if s],
            "risques": [r for r in result["risques"] if r],
            "traitements": [t for t in result["traitements"] if t]
        }

    def get_info_for_condition(self, diagnosis: str):
        return self.get_info(diagnosis)
