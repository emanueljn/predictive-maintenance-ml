import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Leitura dos dados pré-processados
X_train = pd.read_csv('../data/processed/X_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')

y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

# Dicionário para armazenar tempos
tempos = {}

# Implementação do algoritmo de Regressão Logística

parametros_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear']
}

inicio = time.perf_counter()

grid_lr = GridSearchCV(
    LogisticRegression(
        random_state=42,
        max_iter=1000
    ),
    parametros_lr,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)

modelo_lr = grid_lr.best_estimator_

fim = time.perf_counter()

tempo_treino_lr = fim - inicio


inicio = time.perf_counter()

pred_lr = modelo_lr.predict(X_test)

fim = time.perf_counter()

tempo_pred_lr = fim - inicio

tempos['Logistic Regression'] = {
    'Treinamento + GridSearch': tempo_treino_lr,
    'Predição': tempo_pred_lr
}

print("\nMelhores parâmetros Logistic Regression:")
print(grid_lr.best_params_)


#-----------------------------------------------------------------
# Implementação do algoritmo Random Forest

parametros_rf = {
    'n_estimators': [50,100],
    'max_depth': [5,10,15]
}

inicio = time.perf_counter()

grid_rf = GridSearchCV(
    RandomForestClassifier(
        criterion='gini',
        random_state=42
    ),
    parametros_rf,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid_rf.fit(X_train,y_train)

modelo_rf = grid_rf.best_estimator_

fim = time.perf_counter()

tempo_treino_rf = fim - inicio


inicio = time.perf_counter()

pred_rf = modelo_rf.predict(X_test)

fim = time.perf_counter()

tempo_pred_rf = fim - inicio

tempos['Random Forest'] = {
    'Treinamento + GridSearch': tempo_treino_rf,
    'Predição': tempo_pred_rf
}

print("\nMelhores parâmetros Random Forest:")
print(grid_rf.best_params_)

#-----------------------------------------------------------------
# Implementação do algoritmo MLP

parametros_mlp = {
    'hidden_layer_sizes': [
        (16,),
        (32,),
        (64,)
    ],
    'activation': [
        'relu',
        'tanh'
    ],
    'max_iter': [
        300,
        500
    ]
}

inicio = time.perf_counter()

grid_mlp = GridSearchCV(
    MLPClassifier(
        random_state=42
    ),
    parametros_mlp,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid_mlp.fit(
    X_train,
    y_train
)

modelo_mlp = grid_mlp.best_estimator_

fim = time.perf_counter()

tempo_treino_mlp = fim - inicio


inicio = time.perf_counter()

pred_mlp = modelo_mlp.predict(X_test)

fim = time.perf_counter()

tempo_pred_mlp = fim - inicio

tempos['MLP'] = {
    'Treinamento + GridSearch': tempo_treino_mlp,
    'Predição': tempo_pred_mlp
}

print("\nMelhores parâmetros MLP:")
print(grid_mlp.best_params_)

#-----------------------------------------------------------------
# EXPORTAÇÃO DAS VARIÁVEIS
modelos = {
    'Logistic Regression': pred_lr,
    'Random Forest': pred_rf,
    'MLP': pred_mlp
}

joblib.dump(
    modelo_rf,
    '../models/random_forest.pkl'
)

print("Treinamento concluído com sucesso!")

#-----------------------------------------------------------------
# EXIBIÇÃO DOS TEMPOS

print('\n===== TEMPOS DE PROCESSAMENTO =====')

for modelo, dados in tempos.items():

    print(f'\n{modelo}')

    for etapa, valor in dados.items():

        print(
            f'{etapa}: {valor:.4f} segundos'
        )

print("\nTreinamento concluído com sucesso!")