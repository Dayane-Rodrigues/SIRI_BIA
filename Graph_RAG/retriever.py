import os
import logging
import sys
import openai
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_index.core import KnowledgeGraphIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices import PropertyGraphIndex

# Carregar variáveis de ambiente
load_dotenv()

class Retriever:
    def __init__(self):
        # Configurações a partir do .env
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_url = os.getenv("NEO4J_URL")
        self.neo4j_db = 'neo4j'

        # Configurar OpenAI
        openai.api_key = self.openai_api_key
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # Configurar LLM
        self.llm = OpenAI(
            temperature=0, 
            api_key=self.openai_api_key,
            model="gpt-4o-mini"
        )
        Settings.llm = self.llm

        # Conectar ao Neo4j
        self.graph_store = Neo4jPropertyGraphStore(
            username=self.neo4j_username,
            password=self.neo4j_password,
            url=self.neo4j_url,
            database=self.neo4j_db
        )

        # Carregar índice
        self.index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            embed_kg_nodes=True,
            llm=self.llm,
            embed_model=OpenAIEmbedding(model="text-embedding-3-small")
        )

    def retrieve(self, question, top_k=5):
        response = self.index.as_retriever(similarity_top_k=top_k).retrieve(question)
        
        retrieved_texts = [node.node.text for node in response][-top_k:]
        retrieved_scores = [node.score for node in response][-top_k:]

        return retrieved_texts, retrieved_scores

# Exemplo de uso
if __name__ == "__main__":
    retriever = Retriever()
    question = "O curso de IA tem 36 matérias?"
    texts, scores = retriever.retrieve(question)
    print("Textos Recuperados:", texts)
    print("Scores:", scores)
