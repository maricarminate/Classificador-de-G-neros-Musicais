"""
02_train_model.py
Treinamento do modelo de classificação de gêneros musicais
"""

import os
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("=" * 80)
print("🎵 TREINAMENTO DO MODELO - MUSIC GENRE CLASSIFIER")
print("=" * 80)

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

DATA_PATH = 'data/processed'
MODEL_PATH = 'models'
RESULTS_PATH = 'results'

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

print(f"\n⚙️  Configurações:")
print(f"   Épocas: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")

# ============================================================================
# CARREGAR DADOS
# ============================================================================

print("\n[1/5] Carregando dados...")

X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
X_val = np.load(os.path.join(DATA_PATH, 'X_val.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_val.npy'))

# Carregar info
with open(os.path.join(DATA_PATH, 'dataset_info.json'), 'r') as f:
    info = json.load(f)

n_features = info['n_features']
n_classes = info['n_classes']
genres = info['genres']

print(f"   ✅ Treino: {X_train.shape}")
print(f"   ✅ Validação: {X_val.shape}")
print(f"   ✅ Features: {n_features}")
print(f"   ✅ Classes: {n_classes}")

# ============================================================================
# CRIAR MODELO
# ============================================================================

print("\n[2/5] Criando modelo de rede neural...")

modelo = keras.Sequential([
    # Input layer
    layers.Input(shape=(n_features,)),
    
    # Hidden layers
    layers.Dense(512, activation='relu', 
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(256, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(n_classes, activation='softmax')
], name='MusicGenreClassifier')

# Compilar
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Arquitetura do Modelo:")
modelo.summary()

# ============================================================================
# CALLBACKS
# ============================================================================

print("\n[3/5] Configurando callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODEL_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("   ✅ EarlyStopping (patience=15)")
print("   ✅ ReduceLROnPlateau (factor=0.5)")
print("   ✅ ModelCheckpoint (best model)")

# ============================================================================
# TREINAR
# ============================================================================

print("\n[4/5] Iniciando treinamento...")
print("=" * 80)

historico = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 80)
print("✅ TREINAMENTO CONCLUÍDO!")
print("=" * 80)

# ============================================================================
# AVALIAR
# ============================================================================

print("\n[5/5] Avaliando modelo...")

# Predições
y_pred = np.argmax(modelo.predict(X_val), axis=1)

# Métricas
accuracy = np.mean(y_pred == y_val)
print(f"\n📊 Acurácia final: {accuracy*100:.2f}%")

# Relatório de classificação
print("\n📋 Relatório por Gênero:")
print(classification_report(y_val, y_pred, target_names=genres))

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

print("\n📊 Gerando visualizações...")

# 1. Histórico de Treinamento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Acurácia
epochs_range = range(1, len(historico.history['accuracy']) + 1)
ax1.plot(epochs_range, historico.history['accuracy'], 'b-o', 
         label='Treino', linewidth=2.5, markersize=8)
ax1.plot(epochs_range, historico.history['val_accuracy'], 'r-o', 
         label='Validação', linewidth=2.5, markersize=8)
ax1.set_title('🎯 Acurácia do Modelo', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Época', fontsize=14)
ax1.set_ylabel('Acurácia', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Loss
ax2.plot(epochs_range, historico.history['loss'], 'b-o', 
         label='Treino', linewidth=2.5, markersize=8)
ax2.plot(epochs_range, historico.history['val_loss'], 'r-o', 
         label='Validação', linewidth=2.5, markersize=8)
ax2.set_title('📉 Loss do Modelo', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Época', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'), 
            dpi=300, bbox_inches='tight')
print("   ✅ training_history.png")

# 2. Matriz de Confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=genres, yticklabels=genres,
            cbar_kws={'label': 'Frequência'})
plt.title('🎵 Matriz de Confusão - Gêneros Musicais', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predito', fontsize=14)
plt.ylabel('Real', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), 
            dpi=300, bbox_inches='tight')
print("   ✅ confusion_matrix.png")

plt.close('all')

# ============================================================================
# SALVAR MODELO E METADADOS
# ============================================================================

print("\n💾 Salvando modelo...")

# Salvar modelo final
modelo.save(os.path.join(MODEL_PATH, 'music_genre_classifier.h5'))
print("   ✅ music_genre_classifier.h5")

# Salvar histórico
historico_dict = {
    'accuracy': [float(x) for x in historico.history['accuracy']],
    'val_accuracy': [float(x) for x in historico.history['val_accuracy']],
    'loss': [float(x) for x in historico.history['loss']],
    'val_loss': [float(x) for x in historico.history['val_loss']]
}

with open(os.path.join(RESULTS_PATH, 'training_history.json'), 'w') as f:
    json.dump(historico_dict, f, indent=2)
print("   ✅ training_history.json")

# Salvar métricas
metricas = {
    'accuracy': float(accuracy),
    'epochs_trained': len(historico.history['accuracy']),
    'best_val_accuracy': float(max(historico.history['val_accuracy'])),
    'best_val_loss': float(min(historico.history['val_loss'])),
    'genres': genres,
    'n_classes': n_classes,
    'n_features': n_features
}

with open(os.path.join(RESULTS_PATH, 'metrics.json'), 'w') as f:
    json.dump(metricas, f, indent=2)
print("   ✅ metrics.json")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("🎉 MODELO TREINADO COM SUCESSO!")
print("=" * 80)

print(f"\n📊 RESULTADOS:")
print(f"   Acurácia Final: {accuracy*100:.2f}%")
print(f"   Melhor Val Acc: {max(historico.history['val_accuracy'])*100:.2f}%")
print(f"   Épocas: {len(historico.history['accuracy'])}")

print(f"\n📁 ARQUIVOS CRIADOS:")
print(f"   models/music_genre_classifier.h5")
print(f"   models/best_model.h5")
print(f"   results/training_history.png")
print(f"   results/confusion_matrix.png")
print(f"   results/metrics.json")

print(f"\n🎯 PERFORMANCE POR GÊNERO:")
for i, genre in enumerate(genres):
    mask = y_val == i
    if mask.sum() > 0:
        acc_genre = np.mean(y_pred[mask] == y_val[mask])
        emoji_map = {
            'blues': '🎸', 'classical': '🎻', 'country': '🤠',
            'disco': '🕺', 'hiphop': '🎤', 'jazz': '🎷',
            'metal': '🤘', 'pop': '🎤', 'reggae': '🇯🇲', 'rock': '🎸'
        }
        emoji = emoji_map.get(genre, '🎵')
        print(f"   {emoji} {genre:12} {acc_genre*100:5.1f}%")

print(f"\n📝 PRÓXIMO PASSO:")
print(f"   python src/04_classify_music.py sua_musica.mp3")
print(f"   ou")
print(f"   streamlit run web/app_streamlit.py")

print("\n" + "=" * 80)