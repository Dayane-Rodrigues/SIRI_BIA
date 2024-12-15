from fastapi import FastAPI, HTTPException
from retriever_generation import Retriever_Generation

# Inicializa o FastAPI
app = FastAPI()

# Cria uma instância da classe de recuperação
retriever = Retriever_Generation()

@app.get("/")
def home():
    """
    Endpoint básico para verificar se o serviço está funcionando.
    """
    return {"message": "Bem-vindo ao serviço de recuperação e geração de respostas!"}

@app.post("/ask")
def ask_question(question: str):
    """
    Endpoint para perguntar e obter uma resposta.
    """
    try:
        response = retriever.ask_question(question)
        return {"question": question, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)