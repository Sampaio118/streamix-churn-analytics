import pandas as pd
import numpy as np
import os

def gerar_base_streamix():
    np.random.seed(42)
    n_amostras = 1000

    # 1. Perfil dos usuários
    ids = [f"USR_{i:04d}" for i in range(1, n_amostras + 1)]
    
    # Foco maior no público jovem (18 a 29 anos) conforme o desafio
    idades = np.random.choice(
        [np.random.randint(18, 30), np.random.randint(30, 60)], 
        size=n_amostras, 
        p=[0.6, 0.4]
    )
    
    planos = np.random.choice(['Basico', 'Padrao', 'Premium'], size=n_amostras, p=[0.4, 0.4, 0.2])
    localizacoes = np.random.choice(['Sudeste', 'Nordeste', 'Sul', 'Centro-Oeste', 'Norte'], size=n_amostras)

    # 2. Histórico de visualização e interações
    tempo_exibicao_medio_min = np.random.normal(loc=45, scale=15, size=n_amostras).clip(5, 120)
    frequencia_acesso_semana = np.random.poisson(lam=3, size=n_amostras).clip(0, 14)
    dias_ultimo_acesso = np.random.exponential(scale=5, size=n_amostras).astype(int)
    interrupcoes_frequentes = np.random.choice([0, 1], size=n_amostras, p=[0.7, 0.3])
    avaliacoes_media = np.random.uniform(1.0, 5.0, size=n_amostras).round(1)

    # 3. Lógica para definir o Churn (Cancelamento)
    # Usuários jovens, com baixa frequência e muitos dias sem acesso têm mais chance de cancelar
    prob_churn = (
        (idades < 30) * 0.35 +
        (dias_ultimo_acesso > 7) * 0.30 +
        (frequencia_acesso_semana < 2) * 0.25 +
        interrupcoes_frequentes * 0.10
    )
    prob_churn = prob_churn / prob_churn.max() # Normaliza entre 0 e 1
    churn = (np.random.random(n_amostras) < prob_churn).astype(int)

    # Montando o DataFrame
    df = pd.DataFrame({
        'user_id': ids,
        'idade': idades,
        'plano': planos,
        'localizacao': localizacoes,
        'tempo_exibicao_medio_min': tempo_exibicao_medio_min.round(1),
        'frequencia_acesso_semana': frequencia_acesso_semana,
        'dias_ultimo_acesso': dias_ultimo_acesso,
        'interrupcoes_frequentes': interrupcoes_frequentes,
        'avaliacoes_media': avaliacoes_media,
        'churn': churn
    })

    # Garante que a pasta data existe
    os.makedirs("../data", exist_ok=True)
    
    # Caminho para salvar o arquivo na pasta data
    caminho_arquivo = os.path.join(os.path.dirname(__file__), '../data/clientes_streamix.csv')
    df.to_csv(caminho_arquivo, index=False)
    print(f"Sucesso! Base de dados gerada em: {caminho_arquivo}")
    print(f"Total de registros: {n_amostras} | Taxa de Churn simulada: {churn.mean()*100:.1f}%")

if __name__ == "__main__":
    gerar_base_streamix()