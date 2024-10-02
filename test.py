# test.py
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_val, y_val):
    # Avaliar o modelo no conjunto de validação
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Acurácia no conjunto de validação: {val_accuracy * 100:.2f}%')

    # Fazer previsões no conjunto de validação
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Gerar a tabela de confusão
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    return conf_matrix
