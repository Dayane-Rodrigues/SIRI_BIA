import os
import logging
import sys
import openai
from neo4j import GraphDatabase
from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import nest_asyncio
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


# Configurações necessárias
OPENAI_API_KEY = ""
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = ""
NEO4J_URL = ""
NEO4J_DB = "neo4j"
FOLDER_PATH = "/teamspace/studios/this_studio/SIRI_BIA/Dados"
TEMPLATE_PATH = "/teamspace/studios/this_studio/SIRI_BIA/template.txt"
PERSIST_DIR = "/teamspace/studios/this_studio/SIRI_BIA/Graph_RAG"

# Configurar ambiente e OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Configurar LLM
llm = OpenAI(
    temperature=0, 
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"
)
Settings.llm = llm
#Settings.chunk_size = 512

# Conecta ao Neo4j e cria o graph_store
graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URL,
    database=NEO4J_DB
)

# Cria o storage_context a partir do graph_store
#storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Carrega o índice a partir do storage_context já existente
# Supondo que o índice foi criado anteriormente e está armazenado no graph_store.
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_kg_nodes=True,
    llm=llm,
    embed_model= OpenAIEmbedding(model="text-embedding-3-small")
)

# Cria o mecanismo de consulta
query_engine = index.as_query_engine(
    include_text=False,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=3
)

#nest_asyncio.apply()

# Faz a pergunta
pergunta = "O curso de IA tem 36 matérias?"
resposta = query_engine.query(pergunta)
response = index.as_retriever(similarity_top_k=3).retrieve(pergunta)
print(response)

print('*******************************************************')

#kg_rel_texts = [node.node.metadata.get('kg_rel_texts', None) for node in response if node.node.metadata.get('kg_rel_texts', None) is not None]
#kg_rel_map = [node.node.metadata.get('kg_rel_map', None) for node in response if node.node.metadata.get('kg_rel_map', None) is not None]

# Exibe a resposta
#print(f"Pergunta: {pergunta}")
#print(f"Resposta: {resposta}")
#print(kg_rel_texts, kg_rel_map)

# Inicializa listas para armazenar as informações
retrieved_texts = []
retrieved_scores = []
retrieved_relations = []

# Itera sobre o response para extrair os dados necessários
for node_with_score in response:
    node = node_with_score.node
    retrieved_texts.append(node.text)  # Adiciona o texto do nó
    retrieved_scores.append(node_with_score.score)  # Adiciona o score do nó
    
    # Extrai relações, se existirem, a partir dos metadados
    relations = node.metadata.get('kg_rel_texts', None)
    if relations:
        retrieved_relations.append(relations)

retrieved_texts = retrieved_texts[-5:]  
retrieved_scores = retrieved_scores[-5:]  
#retrieved_relations = retrieved_relations[-5:]


# Imprime as listas resultantes
#print("Textos Recuperados:")
#print(retrieved_texts)

#print(len(retrieved_texts))

#print("\nScores:")
#print(retrieved_scores)