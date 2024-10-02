from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    # Early stopping para parar o treinamento se a validação não melhorar
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    # Salvar o melhor modelo
    
    callback = [ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)]
    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callback)  # Adicionando callbacks
    
    return history
