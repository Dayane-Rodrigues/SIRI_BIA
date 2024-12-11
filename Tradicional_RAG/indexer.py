import os
import qdrant_client
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class Indexer:
    """
    Classe responsável por gerenciar a ingestão de dados para um armazenamento vetorial usando Qdrant e OpenAI Embedding.
    Ela carrega documentos de uma pasta especificada, aplica transformações e armazena vetores resultantes no Qdrant.
    """
    def __init__(self, collection_name, data_folder):
        # Configura variáveis de ambiente
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_qdrant = os.getenv("QDRANT_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")

        if not all([self.api_key_openai, self.api_key_qdrant, self.qdrant_url]):
            raise ValueError("Verifique se todas as variáveis de ambiente estão configuradas corretamente no arquivo .env.")

        # Configura o cliente Qdrant
        self.client = qdrant_client.QdrantClient(
            url=self.qdrant_url, 
            api_key=self.api_key_qdrant,
        )

        # Configura o armazenamento vetorial
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=collection_name, 
            enable_hybrid=False
        )

        # Configura o pipeline de ingestão
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=256, chunk_overlap=32),
                OpenAIEmbedding(model='text-embedding-3-small', dimensions=1536)
            ],
            vector_store=self.vector_store,
        )

        # Configura a pasta de dados
        self.data_folder = os.path.abspath(data_folder)

    def load_documents(self):
        """Carrega documentos da pasta especificada."""
        reader = SimpleDirectoryReader(self.data_folder)
        documents = reader.load_data()
        return documents

    def run_pipeline(self):
        """Executa o pipeline de ingestão com os documentos carregados."""
        documents = self.load_documents()
        self.pipeline.run(documents=documents)
        print("Ingestão concluída com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    ingestion_manager = Indexer(
        collection_name="pln_",
        data_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dados")
    )

    ingestion_manager.run_pipeline()
