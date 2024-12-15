FROM python:3.10

# Define o diretório de trabalho no contêiner
WORKDIR /app

# Copia os arquivos do projeto para o contêiner
COPY . /app

# Instala as dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta padrão da API
EXPOSE 8080

# Define o comando padrão para iniciar a aplicação
CMD ["python", "main.py"]
