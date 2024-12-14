import logging
import os
import qdrant_client
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

class Retriever_Generation:
    """
    Uma classe que integra as APIs Qdrant e OpenAI para recuperar e processar consultas.

    Esta classe carrega variáveis de ambiente de um arquivo .env, inicializa conexões com o Qdrant e OpenAI, 
    e permite consultar um repositório vetorial usando modelos incorporados. Ela formata prompts usando um 
    template predefinido e os envia para a API de conclusão de chat do OpenAI para geração de respostas.

    Métodos:
        - __init__: Inicializa os clientes da API e carrega o arquivo de template.
        - _load_template: Lê o template do prompt de um arquivo.
        - ask_question: Recupera textos relevantes, formata o prompt e consulta a API do OpenAI.
    """

    def __init__(self):
        load_dotenv()
        self.client_openai = OpenAI()
        self.client_qdrant = qdrant_client.QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small", dimensions=1536
        )
        self.vector_store = QdrantVectorStore(
            client=self.client_qdrant, collection_name="pln"
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embed_model
        )
        self.template = self._load_template()
        
    def _load_template(self):
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "template.txt")

        with open(template_dir, 'r', encoding='utf-8') as file:
            template = file.read()
            return template

    def retrieve(self, question: str):
        response = self.index.as_retriever(similarity_top_k=5).retrieve(question)
        
        # Extraindo textos e scores
        retrieved_texts = [node.node.text for node in response]
        scores = [node.score for node in response]
        
        return retrieved_texts, scores


    def ask_question(self, question: str):
        response = self.index.as_retriever(similarity_top_k=3).retrieve(question)
        retrieved_texts = [node.node.text for node in response]
        context = "\n".join(retrieved_texts)
        print(question)
        formatted_prompt = self.template.format(context=context, question=question)
        
        completion = self.client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado no BIA."},
                {"role": "user", "content": formatted_prompt}
            ],
        )
        return completion.choices[0].message.content


# Exemplo de uso
if __name__ == "__main__":
    query_engine = Retriever_Generation()
    pergunta = "Dá para trabalhar enquanto estuda? Existem projetos pagos?"
    resposta = query_engine.ask_question(pergunta)
    print(resposta)
