import logging
import sys
import qdrant_client
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
import os


os.environ["OPENAI_API_KEY"] = ""

client = qdrant_client.QdrantClient(
    url="https://dbe31c3d-1494-4b1c-ad8f-4e76bb454a3e.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="",
)

vector_store = QdrantVectorStore(client=client, collection_name="pln_bia", enable_hybrid=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("Qual?")

display(Markdown(f"{response}"))