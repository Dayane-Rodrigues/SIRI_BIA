from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import os

key = "minha_chave"

os.environ["OPENAI_API_KEY"] = key

# define LLM
llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.llm = llm
Settings.chunk_size = 512

# Criar o template do prompt
def create_prompt_template():
    return """
Você é um juiz imparcial encarregado de avaliar a relevância de documentos recuperados para uma consulta de busca.
Siga as instruções abaixo para cada consulta e documentos fornecidos:

1. Analise cuidadosamente o significado e o contexto da consulta.
2. Compare o texto da consulta com cada documento recuperado e classifique sua relevância.
3. Classifique cada documento em uma das categorias abaixo e atribua a pontuação correspondente:
   - Baixa Relevância: O documento menciona o tema da consulta, mas não responde diretamente. Nota = 0.0.
   - Relevância Média: O documento contém informações úteis relacionadas à consulta, mas não responde completamente. Nota = 0.5.
   - Alta Relevância: O documento responde diretamente e de maneira completa à consulta. Nota = 1.0.
4. Use critérios consistentes para todas as consultas.
5. Retorne uma lista de notas para os 5 documentos na sequência em que foram fornecidos.

Entrada:
Consulta: {query}
Documentos Recuperados:
1. {doc1}
2. {doc2}
3. {doc3}
4. {doc4}
5. {doc5}

Saída esperada: [nota1, nota2, nota3, nota4, nota5]
"""

# Criar um objeto de PromptTemplate
prompt_template = PromptTemplate(template=create_prompt_template())

# Exemplo de entrada
query = "Qual é a capital do Brasil?"
documents = [
    "Brasília é a capital do Brasil.",
    "São Paulo é uma das maiores cidades do Brasil.",
    "O Brasil é o maior país da América do Sul.",
    "Brasília foi fundada em 1960.",
    "A economia brasileira é uma das maiores do mundo.",
]

# Formatar os dados para o template
formatted_input = {
    "query": query,
    "doc1": documents[0],
    "doc2": documents[1],
    "doc3": documents[2],
    "doc4": documents[3],
    "doc5": documents[4],
}

# Gerar a avaliação com o LLM
formatted_prompt = prompt_template.format(**formatted_input)
response = llm.complete(formatted_prompt)

# Exibir o resultado
print(response)
