# arquivo: knowledge_graph_indexer.py
import os
import time
import logging
import sys
import openai
from neo4j import GraphDatabase
from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core import StorageContext, load_index_from_storage



class KnowledgeGraphIndexer:
    def __init__(
        self,
        openai_api_key: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_url: str,
        neo4j_database: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        chunk_size: int = 512,
        max_triplets_per_chunk: int = 4,
        include_embeddings: bool = True,
    ):
        self.openai_api_key = openai_api_key
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_url = neo4j_url
        self.neo4j_database = neo4j_database
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.include_embeddings = include_embeddings

        self.template = None  # Será carregado posteriormente

        # Configurações iniciais
        self._configure_environment()
        self._configure_llm()

        # Inicializar atributos que serão preenchidos posteriormente
        self.graph_store = None
        self.index = None
        self.query_engine = None

    def _configure_environment(self):
        # Configura variável de ambiente para a OpenAI
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        # Configura logging
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        # Configura openai
        openai.api_key = self.openai_api_key

    def _configure_llm(self):
        # Define o LLM e as configurações globais
        llm = OpenAI(
            temperature=self.temperature,
            api_key=self.openai_api_key,
            model=self.model
        )
        Settings.llm = llm
        Settings.chunk_size = self.chunk_size

    def connect_graph_store(self):
        """Conecta ao Neo4j e cria a instância de graph_store."""
        self.graph_store = Neo4jPropertyGraphStore(
            username=self.neo4j_username,
            password=self.neo4j_password,
            url=self.neo4j_url,
            database=self.neo4j_database,
        )

    def load_documents(self, folder_path: str):
        """Carrega arquivos .txt do diretório especificado como documentos."""
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(Document(text=content))
        return documents

    def load_template(self, template_path: str):
        """Carrega o template do prompt a partir de um arquivo."""
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def create_index(self, documents):
        """Cria o KnowledgeGraphIndex a partir dos documentos."""
        if self.graph_store is None:
            raise ValueError("graph_store não foi conectado. Chame connect_graph_store() primeiro.")

        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        self.index = PropertyGraphIndex.from_documents(
                documents,
                property_graph_store=self.graph_store,
                embed_kg_nodes=True,
                max_triplets_per_chunk=4,
                include_embeddings = True,
                chunk_size = 512,
                )

    def persist_index(self, persist_dir: str):
        """Persiste o índice localmente no diretório especificado."""
        if self.index is None:
            raise ValueError("O índice não foi criado. Crie o índice antes de persistir.")
        os.makedirs(persist_dir, exist_ok=True)  # Cria o diretório se ele não existir
        self.index.storage_context.persist(persist_dir=persist_dir)


    def load_index_from_storage(self, persist_dir: str):
        """
        Carrega o índice a partir do diretório persistido.
        """
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"O diretório {persist_dir} não foi encontrado.")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = KnowledgeGraphIndex(storage_context=storage_context)
 

    def create_query_engine(
        self,
        include_text: bool = False,
        response_mode: str = "tree_summarize",
        embedding_mode: str = "hybrid",
        similarity_top_k: int = 5
    ):
        """Cria o engine de consultas a partir do índice."""
        if self.index is None:
            raise ValueError("O índice não foi criado ou carregado. Chame create_index() ou load_index_from_storage() primeiro.")
        self.query_engine = self.index.as_query_engine(
            include_text=include_text,
            response_mode=response_mode,
            embedding_mode=embedding_mode,
            similarity_top_k=similarity_top_k
        )

    def query(self, question: str):
        """Executa uma consulta no índice usando o query_engine."""
        if self.query_engine is None:
            raise ValueError("Query engine não foi criado. Chame create_query_engine() primeiro.")
        start = time.time()
        response = self.query_engine.query(question)
        end = time.time()
        elapsed = end - start
        logging.info(f"Tempo de recuperação+geração: {elapsed:.2f} segundos")
        return response

    def get_metadata_from_response(self, response):
        """Extrai metadados específicos (kg_rel_texts e kg_rel_map) da resposta."""
        kg_rel_texts = [
            node.node.metadata.get('kg_rel_texts', None)
            for node in response.source_nodes
            if node.node.metadata.get('kg_rel_texts', None) is not None
        ]
        kg_rel_map = [
            node.node.metadata.get('kg_rel_map', None)
            for node in response.source_nodes
            if node.node.metadata.get('kg_rel_map', None) is not None
        ]
        return kg_rel_texts, kg_rel_map

    def ask_question(self, question: str, similarity_top_k: int = 3):
        """
        Recupera nós do índice e formata o prompt usando o template.
        Depois envia o prompt formatado para o modelo OpenAI.
        """
        if self.index is None:
            raise ValueError("O índice não foi criado ou carregado. Chame create_index() ou load_index_from_storage() primeiro.")

        if self.template is None:
            raise ValueError("Template não carregado. Chame load_template() primeiro.")

        # Usa o retriever para obter nós relevantes
        response = self.index.as_retriever(similarity_top_k=similarity_top_k).retrieve(question)
        retrieved_texts = [node.node.text for node in response]
        context = "\n".join(retrieved_texts)
        
        # Formata o prompt com o template
        formatted_prompt = self.template.format(context=context, question=question)

        # Método atualizado para a API OpenAI
        completion = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Você é um assistente especializado no BIA."},
                {"role": "user", "content": formatted_prompt}
            ],
        )
        return completion.choices[0].message.content


# Exemplo de uso (no primeiro momento, para criar e armazenar o índice)
if __name__ == "__main__":
    # Configurações (troque pelos seus valores reais)
    OPENAI_API_KEY = ""
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = ""
    NEO4J_URL = ""
    NEO4J_DB = "neo4j"
    FOLDER_PATH = "/teamspace/studios/this_studio/SIRI_BIA/Dados"
    TEMPLATE_PATH = "/teamspace/studios/this_studio/SIRI_BIA/template.txt"
    PERSIST_DIR = "/teamspace/studios/this_studio/SIRI_BIA/Graph_RAG"

    indexer = KnowledgeGraphIndexer(
        openai_api_key=OPENAI_API_KEY,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_url=NEO4J_URL,
        neo4j_database=NEO4J_DB,
        model="gpt-4o-mini",
        temperature=0.0,
        chunk_size=512,
        max_triplets_per_chunk=4,
        include_embeddings=True
    )

    indexer.connect_graph_store()
    docs = indexer.load_documents(FOLDER_PATH)
    indexer.load_template(TEMPLATE_PATH)

    start = time.time()
    indexer.create_index(docs) 
    indexer.persist_index(PERSIST_DIR)
 # Cria e armazena o índice no Neo4j
    end = time.time()
    print(f"Tempo para indexação de {len(docs)} documentos: {(end - start)/60:.2f} minutos")

    # Podendo agora fazer perguntas
    #answer = indexer.ask_question("em qual periodo tem algebra linear?")
    #print(answer)
