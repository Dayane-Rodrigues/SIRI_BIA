import logging
import os
import qdrant_client
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

class Retriever_Generation:
    def __init__(self):
        load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
        self.client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        self.template = self._get_template()  # Obtém o texto do template diretamente

    def _get_template(self):
        """
        Define o texto do template diretamente no código.
        """
        return (
            "Você é um assistente especializado no curso de Bacharelado em Inteligência Artificial (BIA). Seu objetivo é fornecer respostas objetivas, gentis e formais às dúvidas apresentadas, com base no contexto fornecido.\n"

            "Contexto: {context} \n"

            "Pergunta: {question}\n"

            "- Seja objetivo, claro e preciso.\n"
            "- Use uma linguagem formal e educada.\n"
            "- Baseie-se no contexto fornecido para criar respostas completas e informativas.\n"
            "- Se a resposta não estiver claramente presente no contexto, diga que apenas 'Não possuo a responda da pergunta: '(cite a pergunta)'. Estou à disposição para esclarecer outras dúvidas que você possa ter."
        )

    def retrieve(self, question: str):
        response = self.index.as_retriever(similarity_top_k=5).retrieve(question)
        retrieved_texts = [node.node.text for node in response]
        scores = [node.score for node in response]
        return retrieved_texts, scores

    def ask_question(self, question: str):
        response = self.index.as_retriever(similarity_top_k=3).retrieve(question)
        retrieved_texts = [node.node.text for node in response]
        context = "\n".join(retrieved_texts)
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
