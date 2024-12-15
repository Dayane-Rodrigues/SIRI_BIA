from retriever_generation import Retriever_Generation

# Testando a classe Retriever_Generation
if __name__ == "__main__":
    # Inicializa a classe
    retriever = Retriever_Generation()

    # Defina uma pergunta de exemplo
    pergunta = "Quais são os desafios de aprender enquanto trabalha?"

    # Testa o método retrieve
    try:
        print("=== Testando o método 'retrieve' ===")
        textos, scores = retriever.retrieve(pergunta)
        print("Textos recuperados:", textos)
        print("Scores:", scores)
    except Exception as e:
        print(f"Erro ao testar o método 'retrieve': {e}")

    # Testa o método ask_question
    try:
        print("\n=== Testando o método 'ask_question' ===")
        resposta = retriever.ask_question(pergunta)
        print("Resposta gerada:", resposta)
    except Exception as e:
        print(f"Erro ao testar o método 'ask_question': {e}")
