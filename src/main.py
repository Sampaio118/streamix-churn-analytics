from config import THRESHOLD, DB_HOST
from preprocessing import carregar_e_preparar_dados
from clustering import executar_clustering
from predictive_model import treinar_modelo_churn

def main():
    print("==================================================")
    print("   PIPELINE CORPORATIVO DE DADOS - STREAMIX+     ")
    print("==================================================")
    print(f"Conectado ao ambiente seguro (Host: {DB_HOST})")
    print(f"Limite de Confiança (Threshold) configurado: {THRESHOLD}\n")
    
    # 1. Executa o Pré-processamento e Validação Inicial
    print("Etapa 1: Carregando e tratando base de dados...")
    df, X_scaled, y = carregar_e_preparar_dados()
    print(f"-> Total de registros carregados: {len(df)}")
    
    # 2. Executa a Segmentação de Clientes (Clustering)
    print("\nEtapa 2: Executando segmentação comportamental (K-Means)...")
    df_segmentado = executar_clustering(n_clusters=4)
    
    # 3. Treina o Modelo Preditivo de Churn (Árvore de Decisão)
    print("\nEtapa 3: Rodando modelo preditivo de cancelamento...")
    modelo_churn = treinar_modelo_churn()
    
    print("\n==================================================")
    print("   PIPELINE EXECUTADO COM SUCESSO NO VS CODE!     ")
    print("==================================================")

if __name__ == "__main__":
    main()