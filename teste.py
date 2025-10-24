"""
03_evaluate_model.py
Avaliação completa do modelo testando múltiplas músicas
"""

import os
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import librosa
from tensorflow import keras
from tqdm import tqdm
import pandas as pd

print("=" * 80)
print("🎵 AVALIAÇÃO COMPLETA DO MODELO")
print("=" * 80)

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

DATA_PATH = 'data/raw'
MODEL_PATH = 'models/music_genre_classifier.h5'
SCALER_PATH = 'data/processed/scaler.pkl'
ENCODER_PATH = 'data/processed/label_encoder.pkl'
INFO_PATH = 'data/processed/dataset_info.json'
RESULTS_PATH = 'results'

SAMPLE_RATE = 22050
DURATION = 30

os.makedirs(RESULTS_PATH, exist_ok=True)

# ============================================================================
# CARREGAR MODELO E PREPROCESSADORES
# ============================================================================

print("\n[1/5] Carregando modelo e preprocessadores...")

modelo = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

with open(INFO_PATH, 'r') as f:
    info = json.load(f)

genres = info['genres']
n_mfcc = info['n_mfcc']
n_chroma = info['n_chroma']

print(f"   ✅ Modelo carregado!")
print(f"   ✅ Gêneros: {genres}")

# ============================================================================
# FUNÇÕES
# ============================================================================

def extract_features(audio_path):
    """Extrai features de um arquivo de áudio"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item() if tempo.size == 1 else float(tempo[0])
        
        features = np.hstack([
            mfccs_mean.flatten(), 
            mfccs_std.flatten(),
            chroma_mean.flatten(), 
            chroma_std.flatten(),
            [sc_mean, sc_std],
            [sr_mean, sr_std],
            [zcr_mean, zcr_std],
            [float(tempo)]
        ])
        
        return features
    
    except Exception as e:
        print(f"      ❌ Erro: {e}")
        return None

# ============================================================================
# COLETAR TODAS AS MÚSICAS PARA TESTE
# ============================================================================

print("\n[2/5] Coletando músicas para teste...")

all_files = []
all_labels = []

# Selecionar 20 músicas aleatórias de cada gênero
import random
random.seed(42)

for genre_idx, genre in enumerate(genres):
    genre_path = os.path.join(DATA_PATH, genre)
    
    if not os.path.exists(genre_path):
        print(f"   ⚠️  Pasta não encontrada: {genre}")
        continue
    
    files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
    
    # Pegar 20 músicas aleatórias (ou todas se tiver menos)
    n_samples = min(20, len(files))
    selected_files = random.sample(files, n_samples)
    
    for file in selected_files:
        all_files.append(os.path.join(genre_path, file))
        all_labels.append(genre)
    
    print(f"   ✅ {genre:12} {n_samples} músicas selecionadas")

print(f"\n   Total: {len(all_files)} músicas para avaliar")

# ============================================================================
# FAZER PREDIÇÕES EM TODAS AS MÚSICAS
# ============================================================================

print("\n[3/5] Fazendo predições...")

predictions = []
true_labels = []
confidences = []
filenames = []

for audio_file, true_label in tqdm(zip(all_files, all_labels), 
                                   total=len(all_files),
                                   desc="   Processando"):
    
    # Extrair features
    features = extract_features(audio_file)
    
    if features is None:
        continue
    
    # Normalizar
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predição
    pred_probs = modelo.predict(features_scaled, verbose=0)[0]
    pred_class = np.argmax(pred_probs)
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    confidence = pred_probs[pred_class]
    
    predictions.append(pred_label)
    true_labels.append(true_label)
    confidences.append(confidence)
    filenames.append(os.path.basename(audio_file))

print(f"\n   ✅ {len(predictions)} predições realizadas")

# ============================================================================
# CALCULAR MÉTRICAS
# ============================================================================

print("\n[4/5] Calculando métricas...")

# Acurácia geral
accuracy = accuracy_score(true_labels, predictions)
print(f"\n📊 ACURÁCIA GERAL: {accuracy*100:.2f}%")

# Métricas por classe
precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, predictions, labels=genres, average=None
)

# Matriz de confusão
cm = confusion_matrix(true_labels, predictions, labels=genres)

# Relatório detalhado
print("\n📋 RELATÓRIO POR GÊNERO:")
print("=" * 80)

results_by_genre = []

for i, genre in enumerate(genres):
    emoji_map = {
        'blues': '🎸', 'classical': '🎻', 'country': '🤠',
        'disco': '🕺', 'hiphop': '🎤', 'jazz': '🎷',
        'metal': '🤘', 'pop': '🎤', 'reggae': '🇯🇲', 'rock': '🎸'
    }
    emoji = emoji_map.get(genre, '🎵')
    
    print(f"{emoji} {genre.upper():12}")
    print(f"   Acurácia:  {recall[i]*100:5.1f}%")
    print(f"   Precisão:  {precision[i]*100:5.1f}%")
    print(f"   F1-Score:  {f1[i]*100:5.1f}%")
    print(f"   Amostras:  {support[i]}")
    print()
    
    results_by_genre.append({
        'genre': genre,
        'accuracy': recall[i],
        'precision': precision[i],
        'f1': f1[i],
        'support': support[i]
    })

# ============================================================================
# ANÁLISES ADICIONAIS
# ============================================================================

print("\n📈 ANÁLISES ADICIONAIS:")
print("=" * 80)

# 1. Confiança média por classe
print("\n🎯 Confiança Média por Classe:")
confidence_by_genre = {}
for genre in genres:
    mask = [predictions[i] == genre for i in range(len(predictions))]
    if any(mask):
        avg_conf = np.mean([confidences[i] for i in range(len(confidences)) if mask[i]])
        confidence_by_genre[genre] = avg_conf
        print(f"   {genre:12} {avg_conf*100:5.1f}%")

# 2. Confusões mais comuns
print("\n🔀 Confusões Mais Comuns (Top 5):")
confusion_pairs = []
for i in range(len(genres)):
    for j in range(len(genres)):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append({
                'true': genres[i],
                'predicted': genres[j],
                'count': cm[i][j]
            })

confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
for pair in confusion_pairs[:5]:
    print(f"   {pair['true']:12} → {pair['predicted']:12} ({pair['count']} vezes)")

# 3. Exemplos de acertos e erros
print("\n✅ Exemplos de ACERTOS (confiança > 90%):")
high_conf_correct = [(filenames[i], true_labels[i], predictions[i], confidences[i]) 
                     for i in range(len(predictions)) 
                     if true_labels[i] == predictions[i] and confidences[i] > 0.9]
for file, true, pred, conf in high_conf_correct[:5]:
    print(f"   {file:30} {true:12} ({conf*100:.1f}%)")

print("\n❌ Exemplos de ERROS (alta confiança mas errado):")
high_conf_wrong = [(filenames[i], true_labels[i], predictions[i], confidences[i]) 
                   for i in range(len(predictions)) 
                   if true_labels[i] != predictions[i] and confidences[i] > 0.7]
for file, true, pred, conf in high_conf_wrong[:5]:
    print(f"   {file:30} Real: {true:10} | Previu: {pred:10} ({conf*100:.1f}%)")

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

print("\n[5/5] Gerando visualizações...")

# 1. Matriz de Confusão Detalhada
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=genres, yticklabels=genres,
            cbar_kws={'label': 'Número de Predições'},
            linewidths=0.5, linecolor='gray')
plt.title('🎵 Matriz de Confusão - Avaliação Completa\n', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('\nGênero Predito', fontsize=14, fontweight='bold')
plt.ylabel('Gênero Real\n', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# Adicionar percentuais
for i in range(len(genres)):
    for j in range(len(genres)):
        if support[i] > 0:
            percentage = (cm[i][j] / support[i]) * 100
            if percentage > 0:
                color = 'white' if cm[i][j] > cm.max()/2 else 'black'
                plt.text(j + 0.5, i + 0.7, f'({percentage:.0f}%)', 
                        ha='center', va='center', color=color, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix_detailed.png'), 
            dpi=300, bbox_inches='tight')
print("   ✅ confusion_matrix_detailed.png")

# 2. Métricas por Gênero (Barras)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

x = np.arange(len(genres))
width = 0.6

# Acurácia
colors_acc = ['#ff6b6b' if recall[i] < 0.7 else '#feca57' if recall[i] < 0.85 else '#48dbfb' 
              for i in range(len(genres))]
bars1 = ax1.bar(x, recall * 100, width, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax1.set_title('🎯 Acurácia por Gênero', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(genres, rotation=45, ha='right')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Meta: 80%')
ax1.legend()

# Adicionar valores nas barras
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Precisão
bars2 = ax2.bar(x, precision * 100, width, color='#ff9ff3', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Precisão (%)', fontsize=12, fontweight='bold')
ax2.set_title('✨ Precisão por Gênero', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(genres, rotation=45, ha='right')
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# F1-Score
bars3 = ax3.bar(x, f1 * 100, width, color='#54a0ff', alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('⚖️ F1-Score por Gênero', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(genres, rotation=45, ha='right')
ax3.set_ylim([0, 100])
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle(f'📊 Métricas de Performance - Acurácia Geral: {accuracy*100:.1f}%',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'metrics_by_genre.png'), 
            dpi=300, bbox_inches='tight')
print("   ✅ metrics_by_genre.png")

# 3. Distribuição de Confiança
plt.figure(figsize=(12, 6))

confidence_correct = [confidences[i] for i in range(len(predictions)) 
                     if true_labels[i] == predictions[i]]
confidence_wrong = [confidences[i] for i in range(len(predictions)) 
                   if true_labels[i] != predictions[i]]

plt.hist(confidence_correct, bins=20, alpha=0.7, label='Acertos', color='green', edgecolor='black')
plt.hist(confidence_wrong, bins=20, alpha=0.7, label='Erros', color='red', edgecolor='black')
plt.xlabel('Confiança', fontsize=12, fontweight='bold')
plt.ylabel('Frequência', fontsize=12, fontweight='bold')
plt.title('📊 Distribuição de Confiança das Predições', fontsize=14, fontweight='bold', pad=15)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'confidence_distribution.png'), 
            dpi=300, bbox_inches='tight')
print("   ✅ confidence_distribution.png")

plt.close('all')

# ============================================================================
# SALVAR RESULTADOS EM JSON E CSV
# ============================================================================

print("\n💾 Salvando resultados...")

# Converter results_by_genre para formato JSON-friendly
genres_json = []
for result in results_by_genre:
    genres_json.append({
        'genre': result['genre'],
        'accuracy': float(result['accuracy']),
        'precision': float(result['precision']),
        'f1': float(result['f1']),
        'support': int(result['support'])
    })

# JSON com métricas gerais
results_json = {
    'overall_accuracy': float(accuracy),
    'total_samples': int(len(predictions)),
    'genres': genres_json,
    'average_confidence': float(np.mean(confidences)),
    'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
    'top_confusions': [
        {
            'true': str(pair['true']),
            'predicted': str(pair['predicted']),
            'count': int(pair['count'])
        } for pair in confusion_pairs[:10]
    ]
}

with open(os.path.join(RESULTS_PATH, 'evaluation_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)
print("   ✅ evaluation_results.json")

# CSV com todas as predições
df = pd.DataFrame({
    'filename': filenames,
    'true_label': true_labels,
    'predicted_label': predictions,
    'confidence': confidences,
    'correct': [true_labels[i] == predictions[i] for i in range(len(predictions))]
})
df.to_csv(os.path.join(RESULTS_PATH, 'predictions_detail.csv'), index=False)
print("   ✅ predictions_detail.csv")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("🎉 AVALIAÇÃO CONCLUÍDA!")
print("=" * 80)

print(f"\n📊 RESUMO:")
print(f"   Total de músicas testadas: {len(predictions)}")
print(f"   Acurácia geral: {accuracy*100:.2f}%")
print(f"   Confiança média: {np.mean(confidences)*100:.2f}%")
print(f"   Acertos: {sum([true_labels[i] == predictions[i] for i in range(len(predictions))])}")
print(f"   Erros: {sum([true_labels[i] != predictions[i] for i in range(len(predictions))])}")

print(f"\n🏆 MELHORES GÊNEROS:")
best_genres = sorted(zip(genres, recall), key=lambda x: x[1], reverse=True)[:3]
for i, (genre, acc) in enumerate(best_genres, 1):
    medal = ['🥇', '🥈', '🥉'][i-1]
    print(f"   {medal} {genre:12} {acc*100:.1f}%")

print(f"\n⚠️  GÊNEROS COM DIFICULDADE:")
worst_genres = sorted(zip(genres, recall), key=lambda x: x[1])[:3]
for genre, acc in worst_genres:
    if acc < 0.8:
        print(f"   • {genre:12} {acc*100:.1f}%")

print(f"\n📁 ARQUIVOS GERADOS:")
print(f"   • results/confusion_matrix_detailed.png")
print(f"   • results/metrics_by_genre.png")
print(f"   • results/confidence_distribution.png")
print(f"   • results/evaluation_results.json")
print(f"   • results/predictions_detail.csv")

print("\n" + "=" * 80)