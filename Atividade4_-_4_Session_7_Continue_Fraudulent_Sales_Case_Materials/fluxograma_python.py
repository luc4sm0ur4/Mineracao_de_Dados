#Lucas Carvalho da Luz Moura
#Matrícula: 2020111816

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- 1. Carregar Dados ---
# Agora lemos o CSV que acabamos de exportar do R.
try:
    df = pd.read_csv('Local_do_CSV/sales_data.csv') #Local onde o CSV foi salvo
except FileNotFoundError:
    print("Erro: Arquivo 'sales_data.csv' não encontrado.")
    print("Certifique-se de executar primeiro o script R (exportar_dados.R) no mesmo diretório.")
    exit()

print("Arquivo 'sales_data.csv' carregado com sucesso.")
print(f"Total de registros: {len(df)}. Total de colunas: {len(df.columns)}")


# --- 2. Engenharia de Características (Função NDTP) ---
# Esta função calcula a estatística NDTP (Distância Normalizada do Preço Típico)
# que o script R identificou como uma importante medida de anomalia [fonte: 4022-4033].
def calculate_ndtp(df):
    # Calcular estatísticas globais (mediana e IQR por produto) dos dados 'ok'
    ok_data = df[df['Insp'] == 'ok']
    
    stats = ok_data.groupby('Prod')['Uprice'].agg(
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    ).reset_index()
    
    stats['iqr'] = stats['q3'] - stats['q1']
    # Evitar divisão por zero, como no script R [fonte: 4030-4032]
    stats.loc[stats['iqr'] == 0, 'iqr'] = stats.loc[stats['iqr'] == 0, 'median']
    # Se a mediana também for 0, definimos o IQR como 1 para evitar divisão por 0.
    stats.loc[stats['iqr'] == 0, 'iqr'] = 1 
    
    stats = stats.drop(columns=['q1', 'q3'])
    
    # Juntar estatísticas de volta ao dataframe principal
    df_merged = df.merge(stats, on='Prod', how='left')
    
    # Calcular NDTP
    df_merged['ndtp'] = np.abs(df_merged['Uprice'] - df_merged['median']) / df_merged['iqr']
    
    # Lidar com produtos não vistos nos dados 'ok' (preencher com 0)
    df_merged['ndtp'] = df_merged['ndtp'].fillna(0)
    # Lidar com valores infinitos caso a mediana e o iqr fossem 0
    df_merged.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_merged.drop(columns=['median', 'iqr'])

print("\nCalculando característica NDTP...")
df_with_ndtp = calculate_ndtp(df)

# --- 3. Preparação para Modelagem Supervisionada ---
# Usar apenas dados rotulados para treinamento supervisionado [fonte: 334]
labeled_df = df_with_ndtp[df_with_ndtp['Insp'].isin(['ok', 'fraud'])].copy()
unlabeled_df = df_with_ndtp[df_with_ndtp['Insp'] == 'unkn'].copy()

# Criar alvo binário (Fraude=1, OK=0)
labeled_df['target'] = labeled_df['Insp'].map({'ok': 0, 'fraud': 1})

# Definir características (X) e alvo (y)
features = ['ID', 'Prod', 'Uprice', 'ndtp'] # Usamos as features do script R + nossa feature NDTP
X = labeled_df[features]
y = labeled_df['target']

print(f"\nTotal de amostras rotuladas para modelagem: {len(y)}")
print(f"Distribuição de classes (0=ok, 1=fraud):\n{y.value_counts()}")

# --- 4. Definir Pipeline de Pré-processamento e Modelagem ---

# Definir colunas categóricas e numéricas
numeric_features = ['Uprice', 'ndtp']
categorical_features = ['ID', 'Prod']

# Criar transformadores
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())]) # Normaliza dados numéricos

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) # Converte texto em números
])

# Combinar transformadores em um único pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='drop') 

# Calcular peso para classe positiva (fraude) para lidar com desequilíbrio [fonte: 340-343]
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"\nCalculado scale_pos_weight: {scale_pos_weight:.2f}")

# Criar o pipeline final: Pré-processamento -> SMOTE -> Classificador
# Usamos ImbPipeline do imbalanced-learn para garantir que o SMOTE [fonte: 3497] só seja aplicado aos dados de treinamento
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # Ponderação adicional para a classe de fraude
        use_label_encoder=False,
        eval_metric='aucpr', # Otimiza para a Área Sob a Curva Precision-Recall
        n_estimators=150,
        random_state=42
    ))
])

# --- 5. Treinamento e Avaliação (Hold-out estratificado) ---
# Dividindo os dados rotulados (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)

print("\nIniciando treinamento do modelo XGBoost com SMOTE...")
model_pipeline.fit(X_train, y_train)
print("Treinamento concluído.")

# Obter probabilidades de previsão para a classe positiva (fraude)
y_probs = model_pipeline.predict_proba(X_test)[:, 1]
y_pred = model_pipeline.predict(X_test)

# Avaliação
ap = average_precision_score(y_test, y_probs)
print(f"\n--- Avaliação no Conjunto de Teste (Hold-out) ---")
print(f"Pontuação de Precisão Média (AUC-PR): {ap:.4f}")
print("\nRelatório de Classificação (com limiar padrão 0.5):")
# Nota: Este relatório de classificação usa um limiar padrão (0.5), que não é ideal para
# ranqueamento, mas nos dá uma ideia do desempenho geral do recall de fraude.
print(classification_report(y_test, y_pred, target_names=['OK', 'Fraud']))

# --- 6. Geração do Ranking Final (O objetivo da tarefa) ---
print("\nGerando ranking de fraude para dados 'unkn'...")
X_unlabeled = unlabeled_df[features].copy()

# Usar o pipeline treinado para prever as probabilidades nos dados não rotulados
unlabeled_probs_final = model_pipeline.predict_proba(X_unlabeled)[:, 1]

# Adicionar probabilidades ao dataframe 'unkn' e ordenar
unlabeled_df['fraud_probability'] = unlabeled_probs_final
inspection_ranking = unlabeled_df.sort_values(by='fraud_probability', ascending=False)

print("\n--- Top 20 Transações Mais Suspeitas (Ranking Final) ---")
# Exibindo as colunas relevantes
print(inspection_ranking[['ID', 'Prod', 'Uprice', 'Insp', 'fraud_probability']].head(20))

# Salvando os resultados
inspection_ranking.to_csv('fraud_ranking_output.csv', index=False)
print("\nRanking completo salvo em 'fraud_ranking_output.csv'")

# --- 7. Plotar Curva PR (Visualização da Avaliação) ---
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.', label='XGBoost + SMOTE')
plt.xlabel('Recall (Sensibilidade)')
plt.ylabel('Precision (Valor Preditivo Positivo)')
plt.title(f'Curva Precision-Recall (AUC-PR = {ap:.3f})')
plt.legend()
plt.grid(True)
plt.savefig('pr_curve_python.png')
print("Gráfico da Curva PR salvo como 'pr_curve_python.png'")