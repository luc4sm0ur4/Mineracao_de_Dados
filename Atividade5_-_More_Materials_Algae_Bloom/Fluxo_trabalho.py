import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer # Adicionado make_scorer

# Pega o caminho absoluto para o diretório onde este script está
BASE_DIR = Path(__file__).resolve().parent

# 1. Carregar Dados
# Nomes das colunas conforme o script R
col_names = ['season','size','speed','mxPH','mnO2','Cl','NO3','NH4',
             'oPO4','PO4','Chla','a1','a2','a3','a4','a5','a6','a7']
test_col_names = ['season','size','speed','mxPH','mnO2','Cl','NO3','NH4',
                  'oPO4','PO4','Chla']
algae_names = ['a1','a2','a3','a4','a5','a6','a7']

# Carregando os dados
df_train = pd.read_csv(
    BASE_DIR / 'Analysis.txt',
    header=None,
    names=col_names,
    sep=r'\s+',
    na_values="XXXXXXX"
)

# (Os outros dataframes não são usados neste fluxo, mas o carregamento está correto)
df_test = pd.read_csv(
    BASE_DIR / 'Eval.txt',
    header=None,
    names=test_col_names,
    sep=r'\s+',
    na_values="XXXXXXX"
)
df_sols = pd.read_csv(
    BASE_DIR / 'Sols.txt',
    header=None,
    names=algae_names,
    sep=r'\s+'
)

# 2. Pré-processamento e Imputação
# Removendo linhas com muitos valores ausentes
percent_missing = df_train.isnull().mean(axis=1)
rows_to_drop = percent_missing[percent_missing > 0.2].index
df_train_tratado = df_train.drop(rows_to_drop).reset_index(drop=True)

# 3. Definição do Pipeline de Pré-processamento (MELHORIA 1)
# Esta é a forma mais "sklearn" de fazer
# Não precisamos mais criar o 'df_clean' manualmente

# Define quais colunas são numéricas (preditores) e quais são categóricas
numeric_features = ['mxPH','mnO2','Cl','NO3','NH4','oPO4','PO4','Chla']
categorical_features = ['season', 'size', 'speed']

# Cria um pipeline para features numéricas:
# 1. Preenche valores ausentes (Imputer)
# 2. Normaliza os dados (Scaler) - Importante para Regressão Linear
numeric_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=10)),
    ('scaler', StandardScaler())
])

# Cria um pipeline para features categóricas:
# 1. Aplica One-Hot Encoding
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina os pipelines numérico e categórico em um único transformador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='drop' # Descarta colunas não especificadas (que não sejam as de 'algae')
)

# 4. Definição dos Modelos e Avaliação (MELHORIA 2)

# Define os preditores (X)
X_train = df_train_tratado.drop(columns=algae_names)

# Define os modelos que queremos comparar
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=1234, ccp_alpha=0.01),
    "Random Forest": RandomForestRegressor(random_state=1234, n_estimators=700)
}

# Define a estratégia de validação cruzada
cv_kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

# Define a métrica de avaliação (MELHORIA 4)
def calculate_nmse(y_true, y_pred):
    """Calcula o NMSE (como definido no script R)"""
    mse = np.mean((y_true - y_pred)**2)
    baseline_mse = np.mean((y_true - np.mean(y_true))**2) # Var(y_true)
    # Adiciona proteção contra divisão por zero
    if baseline_mse == 0:
        return 1.0 if mse > 0 else 0.0
    return mse / baseline_mse

# Converte a função em um 'scorer' utilizável pelo sklearn
# 'greater_is_better=False' significa que um valor menor da métrica é melhor
nmse_scorer = make_scorer(calculate_nmse, greater_is_better=False)

# 5. Execução do Loop de Modelagem (MELHORIA 3)
# Dicionário para guardar os resultados finais
all_results = {}

# Loop para cada uma das 7 algas alvo
for target_algae in algae_names:
    print(f"\n==========================================")
    print(f"  Resultados da Modelagem para: {target_algae}")
    print(f"==========================================")
    
    # Pega o 'y' (alvo) correspondente
    y_train = df_train_tratado[target_algae]
    
    # Dicionário para guardar os resultados desta alga
    target_results = {}

    # Loop para cada um dos 3 modelos
    for model_name, model_regressor in models.items():
        
        # Cria o pipeline completo: Processador + Regressor
        full_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_regressor)
        ])
        
        # Executa a validação cruzada
        # (O 'pipe' cuida da imputação e scaling em cada 'fold' separadamente)
        scores = cross_val_score(full_pipe, X_train, y_train, 
                                 cv=cv_kfold, 
                                 scoring=nmse_scorer)
        
        # O scorer retorna valores negativos, então multiplicamos por -1
        mean_nmse = scores.mean() * -1
        target_results[model_name] = mean_nmse
        
        print(f"  -> {model_name}: NMSE Médio = {mean_nmse:.4f}")
    
    all_results[target_algae] = target_results

print("\n--- Fim da Execução ---")

# Você pode inspecionar a variável 'all_results' para ver o placar final
print("\nResumo dos Resultados (NMSE Médio):")
print(pd.DataFrame(all_results).T)