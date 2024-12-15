from llama_index.core import KnowledgeGraphIndex, StorageContext
import os

# Defina a chave da API diretamente no código
os.environ["OPENAI_API_KEY"] = "chave-api"

# Caminho para o diretório onde os arquivos do índice foram persistidos
PERSIST_DIR = "/teamspace/studios/this_studio/SIRI_BIA/Graph_RAG"

# Carregando o índice salvo
def load_index(persist_dir):
    """Carrega o índice KnowledgeGraphIndex a partir do diretório persistido."""
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = KnowledgeGraphIndex(storage_context=storage_context)
    return index

# Configurando o query_engine e fazendo perguntas
def infer_from_index(index, question):
    """Realiza uma inferência no índice carregado."""
    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        similarity_top_k=5
    )
    response = query_engine.query(question)
    return response

if __name__ == "__main__":
    # Carregando o índice
    index = load_index(PERSIST_DIR)
    
    # Fazendo uma pergunta
    question = "Qual é o período que contém Álgebra Linear?"
    response = infer_from_index(index, question)
    
    # Exibindo a resposta
    print("Resposta:", response)
