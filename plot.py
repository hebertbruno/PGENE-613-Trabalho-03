import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy(history):
    # Plotar acurácia

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.title('Acurácia ao longo do treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(f'results/accuracy.png')
    plt.close()

def plot_loss(history):
    #plotaar perda
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda Treinamento')
    plt.plot(history.history['val_loss'], label='Perda Validação')
    plt.title('Perda ao longo do treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.savefig(f'results/loss.png')
    plt.close()
    #plt.show()

def plot_confusion_matrix(conf_matrix, activity_mapping):
    # Obter os nomes das atividades a partir do mapeamento
    #activity_labels = [activity_mapping[i] for i in range(1, len(activity_mapping) + 1)]
    activity_labels = [i for i in range(1, len(activity_mapping) + 1)]

    # Plotar a tabela de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=activity_labels, yticklabels=activity_labels)
    
    plt.title('Tabela de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.savefig(f'results/confusion_matrix.png')
    plt.close()
