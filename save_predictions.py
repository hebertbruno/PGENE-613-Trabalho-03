import pandas as pd
import numpy as np

def save_predictions_to_csv(model, X_val, y_val, activity_mapping, filename='predictions.csv'):
    # Fazer previsões no conjunto de validação
    predictions = model.predict(X_val)

    # Pegar as classes preditas (a classe com maior probabilidade)
    predicted_classes = np.argmax(predictions, axis=1)

    # Pegar as classes reais
    real_classes = np.argmax(y_val, axis=1)

    # Converter os índices preditos e reais para os nomes das atividades (opcional)
    real_activity_labels = [activity_mapping[i+1] for i in real_classes]
    predicted_activity_labels = [activity_mapping[i+1] for i in predicted_classes]

    # Criar um DataFrame com valores reais e preditos
    df = pd.DataFrame({
        'Real': real_activity_labels,
        'Predito': predicted_activity_labels
    })

    # Salvar o DataFrame em um arquivo CSV
    df.to_csv(filename,sep=';', index=False)
    print(f'Resultados salvos em {filename}')
