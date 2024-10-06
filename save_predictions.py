import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def save_predictions_to_csv(model_path, X_val, y_val, activity_mapping, filename='predictions.csv'):
    # Fazer previsões no conjunto de validação
    model = load_model(model_path)
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
