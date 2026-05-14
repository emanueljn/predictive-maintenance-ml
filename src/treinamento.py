import pandas as pd
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

# Implementação do algoritmo de regressão logistica
modelo_lr = LogisticRegression(
    solver='liblinear',
    random_state=42,
    max_iter=1000
)

modelo_lr.fit(X_train, y_train)

pred_lr = modelo_lr.predict(X_test)

#-----------------------------------------------------------------
# Implementação do algoritmo Random Forest
modelo_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    criterion='gini',
    random_state=42
)

modelo_rf.fit(X_train, y_train)

parametros_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15]
}

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

grid_rf.fit(X_train, y_train)

modelo_rf = grid_rf.best_estimator_

pred_rf = modelo_rf.predict(X_test)

#-----------------------------------------------------------------
# Implementação do algoritmo MLP
modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

modelo_mlp.fit(X_train, y_train)

pred_mlp = modelo_mlp.predict(X_test)

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