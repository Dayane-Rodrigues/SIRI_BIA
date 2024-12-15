from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever_generation import Retriever_Generation

# Inicializa a aplicação FastAPI
app = FastAPI()

# Modelo de entrada - define que a API espera um JSON com o campo 'question'
class QuestionRequest(BaseModel):
    question: str

# Instancia a classe Retriever_Generation
retriever = Retriever_Generation()

@app.get("/")
def home():
    """
    Endpoint básico para verificar se o serviço está funcionando.
    """
    return {"message": "Bem-vindo ao serviço de recuperação e geração de respostas!"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Endpoint que recebe uma pergunta no formato JSON e retorna a resposta.
    """
    try:
        # Obtém a pergunta do JSON enviado e gera a resposta
        response = retriever.ask_question(request.question)
        return {"question": request.question, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")

# Executa o servidor no modo direto (para testes locais)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
