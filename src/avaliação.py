from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from treinamento import modelos, y_test

# FUNÇÃO DE AVALIAÇÃO
def avaliar_modelo(nome, y_real, predicoes):

    print(f'\\n===== {nome} =====')

    print('Acurácia:',
          accuracy_score(y_real, predicoes))

    print('Precisão:',
          precision_score(y_real, predicoes))

    print('Recall:',
          recall_score(y_real, predicoes))

    print('F1-score:',
          f1_score(y_real, predicoes))

    print('\\nMatriz de confusão:')

    print(confusion_matrix(y_real, predicoes))

# AVALIAÇÃO DOS MODELOS
for nome, predicoes in modelos.items():
    avaliar_modelo(nome, y_test, predicoes)