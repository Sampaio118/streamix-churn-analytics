import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import carregar_e_preparar_dados
from config import THRESHOLD

def treinar_modelo_churn():
    print("\n=== Treinando Modelo Preditivo de Churn (Árvore de Decisão) ===")
    
    # 1. Carregar dados processados
    df, X_scaled, y = carregar_e_preparar_dados()
    
    if y is None:
        print("Erro: A coluna 'churn' não foi encontrada nos dados.")
        return
    
    # 2. Dividir em base de treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Treinar o modelo de Árvore de Decisão
    # Limitamos a profundidade (max_depth=4) para manter a árvore limpa e fácil de interpretar
    modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)
    
    # 4. Avaliar o modelo
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    print(f"Modelo treinado com sucesso! Acurácia na base de teste: {acuracia * 100:.2f}%")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred))
    
    # 5. Exibir as regras lógicas aprendidas pela árvore
    features_nomes = list(X_scaled.columns)
    regras = export_text(modelo, feature_names=features_nomes)
    print("\nRegras de Decisão da Árvore (O 'Caminho Lógico' do Churn):")
    print(regras)
    
    return modelo

if __name__ == "__main__":
    treinar_modelo_churn()