import json
import os
import csv
import pandas as pd
from typing import List
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from retriever import Retriever_Generation

# Carregue as variáveis de ambiente do arquivo .env
load_dotenv()


class AnswerRelevance:
    def __init__(self, api_key: str, csv_file: str, top_n: int):
        self.api_key = api_key
        self.csv_file = csv_file
        self.top_n = top_n
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.llm = OpenAI(temperature=0, model="gpt-4o-mini")
        Settings.llm = self.llm
        self.retriever = Retriever_Generation()

    @staticmethod
    def load_prompt_template(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def evaluate_documents(self, query: str, documents: List[str]) -> str:
        prompt_template = PromptTemplate(template=self.load_prompt_template('template_answer_relevance.txt'))
        docs_formatted = "\n\n".join([f"Documento {i+1}: {doc}" for i, doc in enumerate(documents)])
        formatted_input = {
            "query": query,
            "docs": docs_formatted,
        }
        formatted_prompt = prompt_template.format(**formatted_input)
        response = self.llm.complete(formatted_prompt)
        return response.text.strip() if hasattr(response, 'text') else str(response)

    def extract_and_save_info(self, query: str):
        texts, scores = self.retriever.retrieve(query)
        llm_judgment = self.evaluate_documents(query, texts)
        
        row_data = {
            "query": query,
            "texts": texts,
            "scores": scores,
            "llm_judgment": llm_judgment
        }

        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = ["query"] + [f"text_{i}" for i in range(1, self.top_n + 1)] + \
                         [f"score_{i}" for i in range(1, self.top_n + 1)] + ["llm_judgment"]
                writer.writerow(header)
            writer.writerow(
                [row_data['query']] + row_data['texts'] + row_data['scores'] + [row_data['llm_judgment']]
            )
        print(f"Informações extraídas e salvas em {self.csv_file}")

    def metrics_calculation(self):
        df = pd.read_csv(self.csv_file)
        not_lists_count = 0
        precision_at_k = []
        mean_reciprocal_rank = []
        similarity_scores = []

        for _, row in df.iterrows():
            llm_judgment = row['llm_judgment']
            if not isinstance(llm_judgment, str) or not llm_judgment.startswith('['):
                not_lists_count += 1
                continue

            judgments = json.loads(llm_judgment)
            if not judgments:
                continue

            precision = judgments.count(1.0) / len(judgments)
            precision_at_k.append(precision)

            max_index = judgments.index(max(judgments))
            result = 1 / (max_index + 1)
            mean_reciprocal_rank.append(result)

            scores = [row[col] for col in df.columns if col.startswith('score_') and pd.notna(row[col])]
            if scores:
                similarity_scores.extend(scores)

        avg_precision = sum(precision_at_k) / len(precision_at_k) if precision_at_k else 0
        avg_mrr = sum(mean_reciprocal_rank) / len(mean_reciprocal_rank) if mean_reciprocal_rank else 0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

        print(f"Resultados das Métricas:\n")
        print(f"Saídas Irregulares: {not_lists_count}\n")
        print(f"Precision@k: {avg_precision:.4f}")
        print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")
        print(f"Average Similarity: {avg_similarity:.4f}")

if __name__ == "__main__":
    csv_file = "answer_relevance_outputs.csv"
    if os.path.isfile(csv_file):
        os.remove(csv_file)
        print(f"Arquivo {csv_file} excluído com sucesso.")

    with open('queries.json', 'r', encoding='utf-8') as file:
        queries = json.load(file)['queries']

    answer_relevance = AnswerRelevance(
        api_key=os.getenv('OPENAI_API_KEY'),
        csv_file=csv_file,
        top_n=5
    )

    for query in queries:
        answer_relevance.extract_and_save_info(query)

    answer_relevance.metrics_calculation()
