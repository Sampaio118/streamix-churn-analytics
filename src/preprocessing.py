import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import DATA_PATH

def carregar_e_preparar_dados():
    # Como estamos na pasta src, ajustamos o caminho se necessário para encontrar a pasta data
    caminho_completo = os.path.join(os.path.dirname(__file__), "..", DATA_PATH)
    
    if not os.path.exists(caminho_completo):
        # Tenta o caminho direto caso venha da raiz
        caminho_completo = DATA_PATH

    print(f"Carregando dados de: {caminho_completo}")
    df = pd.read_csv(caminho_completo)

    # 1. Tratamento de Categóricas (Label Encoding para 'plano' e 'localizacao')
    le_plano = LabelEncoder()
    le_localizacao = LabelEncoder()
    
    df['plano_encoded'] = le_plano.fit_transform(df['plano'])
    df['localizacao_encoded'] = le_localizacao.fit_transform(df['localizacao'])

    # 2. Seleção de features (recursos) para o modelo e clustering
    features = [
        'idade', 'plano_encoded', 'localizacao_encoded', 
        'tempo_exibicao_medio_min', 'frequencia_acesso_semana', 
        'dias_ultimo_acesso', 'interrupcoes_frequentes', 'avaliacoes_media'
    ]
    
    X = df[features]
    y = df['churn'] if 'churn' in df.columns else None

    # 3. Normalização das escalas (StandardScaler - essencial para K-Means e Redes Neurais)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    print("Pré-processamento concluído com sucesso!")
    return df, X_scaled_df, y

if __name__ == "__main__":
    df, X_scaled, y = carregar_e_preparar_dados()
    print("\nPrimeiras linhas dos dados normalizados:")
    print(X_scaled.head())