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
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5
)

#nest_asyncio.apply()

# Faz a pergunta
pergunta = "O curso de IA tem 36 matérias?"
resposta = query_engine.query(pergunta)

# Exibe a resposta
print(f"Pergunta: {pergunta}")
print(f"Resposta: {resposta}")
