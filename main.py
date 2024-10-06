from load_data import load_and_process_data
from model import create_lstm_model
from train import train_model
from test import evaluate_model
from plot import plot_accuracy, plot_loss, plot_confusion_matrix
from save_predictions import save_predictions_to_csv

# Caminhos dos arquivos
data_path = 'adl_data.mat'
labels_path = 'adl_labels.mat'
names_path = 'adl_names.mat'
best_model_path = 'best_model.keras'

# Carregar os dados
X_train, X_val, y_train, y_val, activity_mapping = load_and_process_data(data_path, labels_path, names_path)
# Ajuste dos dados de entrada
X_train = X_train.reshape((X_train.shape[0], 151, 3))  # Transformando para (n_amostras, 151 timesteps, 3 features)
X_val = X_val.reshape((X_val.shape[0], 151, 3))

# Criar o modelo LSTM
input_shape = (151, 3)  # 151 timesteps, 3 features (X, Y, Z)
model = create_lstm_model(input_shape)

# Treinar o modelo
history = train_model(model, X_train, y_train, X_val, y_val)

# Avaliar o modelo e gerar a tabela de confusão
conf_matrix = evaluate_model(best_model_path, X_val, y_val)

# Plotar os gráficos de acurácia e perda
plot_accuracy(history)
plot_loss(history)

# Plotar a tabela de confusão (usando o mapeamento de atividades para mostrar os rótulos compreensíveis)
plot_confusion_matrix(conf_matrix, activity_mapping)

# Salvar as previsões e valores reais em um arquivo CSV
save_predictions_to_csv(best_model_path, X_val, y_val, activity_mapping, 'results/validation_predictions.csv')