import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.treinamento import pred_lr
from treinamento import (
    modelo_rf,
    X_test,
    y_test,
    pred_rf, pred_mlp
)

# AMOSTRAS
dados_teste = X_test.copy()

dados_teste['Falha Real'] = y_test

sem_falha = dados_teste[
    dados_teste['Falha Real'] == 0
].head(3)

com_falha = dados_teste[
    dados_teste['Falha Real'] == 1
].head(2)

amostra_final = pd.concat([
    sem_falha,
    com_falha
])

X_amostra = amostra_final.drop(
    'Falha Real',
    axis=1
)

y_real = amostra_final['Falha Real']

# PREDIÇÕES
predicoes = modelo_rf.predict(X_amostra)

resultado = pd.DataFrame({

    'Máquina': [1,2,3,4,5],

    'Valor Real': y_real.values,

    'Predição': predicoes

})

print(resultado)

# TAXA DE ACERTO
acertos = sum(y_real.values == predicoes)

taxa = acertos / len(y_real)

print(f'\\nTaxa de acerto: {taxa:.2%}')

# MATRIZ DE CONFUSÃO - REGRESSÃO LOGÍSTICA

matriz = confusion_matrix(y_test, pred_lr)

print('\n===== MATRIZ DE CONFUSÃO =====')

print(matriz)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=[
        'Sem Falha',
        'Falha'
    ]
)

disp.plot(cmap='Blues')

plt.show()

# MATRIZ DE CONFUSÃO - RAMDOM FOREST
matriz = confusion_matrix(y_test, pred_rf)

print('\\n===== MATRIZ DE CONFUSÃO =====')

print(matriz)

disp = ConfusionMatrixDisplay(

    confusion_matrix=matriz,

    display_labels=[
        'Sem Falha',
        'Falha'
    ]
)

disp.plot(cmap='Blues')
plt.show()

# MATRIZ DE CONFUSÃO - MULTILAYER PERCEPTRON
matriz = confusion_matrix(y_test, pred_mlp)

print('\n===== MATRIZ DE CONFUSÃO =====')
print(matriz)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=[
        'Sem Falha',
        'Falha'
    ]
)
disp.plot(cmap='Blues')
plt.show()
