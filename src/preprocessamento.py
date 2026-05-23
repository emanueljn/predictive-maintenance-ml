import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# LEITURA DO DATASET
caminho_dataset = '../data/raw/ai4i2020.csv'
df = pd.read_csv(caminho_dataset)

# TRATAMENTO DE DADOS AUSENTES
print('\nValores ausentes antes do tratamento:')
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))

# Remove UDI, Product ID e os tipos específicos de falha para evitar Data Leakage
colunas_para_remover = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

# CODIFICAÇÃO DE VARIÁVEIS
encoder = LabelEncoder()
if 'Type' in df.columns:
    df['Type'] = encoder.fit_transform(df['Type'])

# ANÁLISE E TRATAMENTO DE OUTLIERS
plt.figure(figsize=(12, 6))
df.boxplot()
plt.xticks(rotation=45)
plt.title('Distribuição dos Dados dos Sensores')
plt.tight_layout()
plt.show()

# SELEÇÃO E PREPARAÇÃO DE ATRIBUTOS
y = df['Machine failure']
X = df.drop('Machine failure', axis=1)

print('\nVariáveis preditoras finais (Apenas sensores e Type):')
print(X.columns)

# DIVISÃO DOS DADOS
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print('\nTamanho dos conjuntos:')
print(f'Treino: {X_train.shape}')
print(f'Validação: {X_val.shape}')
print(f'Teste: {X_test.shape}')

# NORMALIZAÇÃO DOS DADOS
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print('\nNormalização concluída.')

# EXPORTAÇÃO DOS DADOS PROCESSADOS
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val_df = pd.DataFrame(X_val_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Salvando arquivos únicos estruturados
import os
os.makedirs('../data/processed', exist_ok=True)

X_train_df.to_csv('../data/processed/X_train.csv', index=False)
X_val_df.to_csv('../data/processed/X_val.csv', index=False)
X_test_df.to_csv('../data/processed/X_test.csv', index=False)

pd.DataFrame(y_train).to_csv('../data/processed/y_train.csv', index=False)
pd.DataFrame(y_val).to_csv('../data/processed/y_val.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/processed/y_test.csv', index=False)

print('\nArquivos processados exportados com sucesso.')