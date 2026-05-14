import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# LEITURA DO DATASET
# Caminho do dataset
caminho_dataset = '../data/raw/ai4i2020.csv'

# Leitura do arquivo CSV
df = pd.read_csv(caminho_dataset)

# TRATAMENTO DE DADOS AUSENTES
print('\nValores ausentes antes do tratamento:')
print(df.isnull().sum())

# Preenchimento de valores nulos utilizando a mediana
df = df.fillna(df.median(numeric_only=True))

print('\nValores ausentes após o tratamento:')
print(df.isnull().sum())


# ANÁLISE E TRATAMENTO DE OUTLIERS
# Visualização inicial utilizando boxplot

plt.figure(figsize=(12, 6))

df.boxplot()

plt.xticks(rotation=45)
plt.title('Análise Inicial de Outliers')
plt.tight_layout()
plt.show()

# Remoção de outliers utilizando IQR
colunas_numericas = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

for coluna in colunas_numericas:

    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    df = df[
        (df[coluna] >= limite_inferior) &
        (df[coluna] <= limite_superior)
    ]

print('\nDataset após remoção de outliers:')
print(df.shape)

# CODIFICAÇÃO DE VARIÁVEIS
# Codificação da variável categórica Type

encoder = LabelEncoder()

if 'Type' in df.columns:
    df['Type'] = encoder.fit_transform(df['Type'])

print('\nValores codificados da variável Type:')
print(df['Type'].head())


# SELEÇÃO E PREPARAÇÃO DE ATRIBUTOS
# Remoção de colunas irrelevantes
if 'UDI' in df.columns:
    df = df.drop('UDI', axis=1)

if 'Product ID' in df.columns:
    df = df.drop('Product ID', axis=1)

# Variável alvo
y = df['Machine failure']

# Variáveis preditoras
X = df.drop('Machine failure', axis=1)

print('\nVariáveis preditoras:')
print(X.columns)

print('\nVariável alvo:')
print(y.head())

# DIVISÃO DOS DADOS
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print('\nTamanho dos conjuntos:')
print(f'Treino: {X_train.shape}')
print(f'Validação: {X_val.shape}')
print(f'Teste: {X_test.shape}')

# NORMALIZAÇÃO DOS DADOS
scaler = MinMaxScaler()

# Ajuste somente no treino
X_train = scaler.fit_transform(X_train)

# Transformação validação e teste
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print('\nNormalização concluída.')

# EXPORTAÇÃO DOS DADOS PROCESSADOS
# Conversão novamente para DataFrame
X_train_df = pd.DataFrame(X_train)
X_val_df = pd.DataFrame(X_val)
X_test_df = pd.DataFrame(X_test)

# Salvando arquivos processados
X_train_df.to_csv('../data/processed/X_train.csv', index=False)
X_val_df.to_csv('../data/processed/X_val.csv', index=False)
X_test_df.to_csv('../data/processed/X_test.csv', index=False)

pd.DataFrame(y_train).to_csv('../data/processed/y_train.csv', index=False)
pd.DataFrame(y_val).to_csv('../data/processed/y_val.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/processed/y_test.csv', index=False)

print('\nArquivos processados exportados com sucesso.')