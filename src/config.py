import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env localizado na raiz do projeto
load_dotenv()

# Mapeando as variáveis de ambiente para variáveis Python
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "default_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.7))
DATA_PATH = os.getenv("DATA_PATH", "data/clientes_streamix.csv")

if __name__ == "__main__":
    print("=== Teste de Configuração ===")
    print(f"Host do Banco: {DB_HOST}")
    print(f"Usuário: {DB_USER}")
    print(f"Limite de Churn (Threshold): {THRESHOLD}")
    print(f"Caminho dos Dados: {DATA_PATH}")