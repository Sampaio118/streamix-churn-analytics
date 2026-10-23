import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import carregar_e_preparar_dados

def executar_clustering(n_clusters=4):
    print("\n=== Executando Segmentação de Clientes (K-Means) ===")
    
    # Carrega os dados pré-processados
    df, X_scaled, y = carregar_e_preparar_dados()
    
    # Aplica o algoritmo K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Adiciona o grupo (cluster) de volta ao DataFrame original
    df['cluster'] = clusters
    
    print(f"Clientes segmentados em {n_clusters} grupos com sucesso!")
    
    # Mostra um resumo do perfil de cada cluster (ex: média de idade e dias sem acesso)
    resumo = df.groupby('cluster')[['idade', 'frequencia_acesso_semana', 'dias_ultimo_acesso', 'churn']].mean()
    print("\nPerfil Médio por Cluster:")
    print(resumo)
    
    return df

if __name__ == "__main__":
    executar_clustering()