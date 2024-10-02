import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_and_process_data(data_path, labels_path, names_path):
    # Carregar os dados do arquivo .mat
    data_contents = scipy.io.loadmat(data_path)
    labels_contents = scipy.io.loadmat(labels_path)
    names_content = scipy.io.loadmat(names_path)
    
    data = data_contents['adl_data']
    labels = labels_contents['adl_labels']
    names = names_content['adl_names'].flatten()  # Mapeamento das atividades

    # Verifique as formas dos arrays
    print(f"Shape de data: {data.shape}")
    print(f"Shape de labels: {labels.shape}")
    print(f"Shape de names: {names.shape}")

    # Normalizar os dados
    data = data / np.max(data)

    # Verifique a consistência dos dados
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"Inconsistência: {data.shape[0]} amostras de dados e {labels.shape[0]} rótulos.")

    # Extrair apenas a primeira coluna de labels para y
    y = labels[:, 0]  # Usar somente a primeira coluna

    # Converter y para uma representação one-hot
    y_one_hot = to_categorical(y - 1, num_classes=len(names))  # Ajustando o rótulo para zero-based

    # Dividir os dados em conjunto de treino e validação
    X_train, X_val, y_train, y_val = train_test_split(data, y_one_hot, test_size=0.15, random_state=42)

    # Criar um mapeamento de rótulos para atividades
    activity_mapping = {i + 1: names[i] for i in range(len(names))}  # Mapear rótulos de 1 a 9

    # Retornar os dados processados
    return X_train, X_val, y_train, y_val, activity_mapping
