"""
01_prepare_data.py
Preparação dos dados: extração de features e criação de espectrogramas
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🎵 PREPARAÇÃO DOS DADOS - GTZAN DATASET")
print("=" * 80)

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

DATA_PATH = 'data/raw'
OUTPUT_PATH = 'data/processed'
SPECTROGRAM_PATH = 'data/spectrograms'

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

SAMPLE_RATE = 22050
DURATION = 30  # segundos
N_MFCC = 13
N_CHROMA = 12

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SPECTROGRAM_PATH, exist_ok=True)

print(f"\n📁 Diretórios:")
print(f"   Input:  {DATA_PATH}")
print(f"   Output: {OUTPUT_PATH}")
print(f"   Specs:  {SPECTROGRAM_PATH}")

# ============================================================================
# FUNÇÕES DE EXTRAÇÃO DE FEATURES
# ============================================================================

def extract_features(audio_path):
    """
    Extrai múltiplas features de um arquivo de áudio
    """
    try:
        # Carregar áudio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs, axis=1)  # Média ao longo do tempo
        mfccs_std = np.std(mfccs, axis=1)    # Desvio padrão ao longo do tempo
        
        # 2. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 3. Spectral Centroid (brilho)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 6. Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Garantir que tempo seja escalar
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item() if tempo.size == 1 else float(tempo[0])
        
        # Concatenar todas as features (garantir arrays 1D)
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
        print(f"   ❌ Erro em {audio_path}: {e}")
        return None

def create_spectrogram(audio_path, output_path):
    """
    Cria e salva o espectrograma de um arquivo de áudio
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Salvar como imagem (sem eixos para CNN)
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"   ❌ Erro no spectrogram {audio_path}: {e}")
        return False

# ============================================================================
# VERIFICAR DATASET
# ============================================================================

print("\n[1/5] Verificando dataset...")

total_files = 0
for genre in GENRES:
    genre_path = os.path.join(DATA_PATH, genre)
    if os.path.exists(genre_path):
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        n_files = len(files)
        total_files += n_files
        print(f"   ✅ {genre:12} {n_files:3} arquivos")
    else:
        print(f"   ❌ {genre:12} não encontrado!")

print(f"\n   Total: {total_files} arquivos")

if total_files == 0:
    print("\n❌ Dataset não encontrado!")
    print("   Execute: python download_dataset.py")
    exit(1)

# ============================================================================
# EXTRAIR FEATURES
# ============================================================================

print("\n[2/5] Extraindo features de áudio...")

all_features = []
all_labels = []
all_filenames = []

for genre in GENRES:
    genre_path = os.path.join(DATA_PATH, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
    
    print(f"\n   🎵 {genre.upper()}:")
    
    for file in tqdm(files, desc=f"   Processando {genre}", ncols=70):
        audio_path = os.path.join(genre_path, file)
        
        # Extrair features
        features = extract_features(audio_path)
        
        if features is not None:
            all_features.append(features)
            all_labels.append(genre)
            all_filenames.append(file)

print(f"\n   ✅ Features extraídas: {len(all_features)} arquivos")

# Converter para arrays NumPy
X = np.array(all_features)
y = np.array(all_labels)

print(f"\n   Shape das features: {X.shape}")
print(f"   Shape dos labels: {y.shape}")

# ============================================================================
# CRIAR ESPECTROGRAMAS
# ============================================================================

print("\n[3/5] Criando espectrogramas...")

for genre in GENRES:
    os.makedirs(os.path.join(SPECTROGRAM_PATH, genre), exist_ok=True)

spec_count = 0

for genre in GENRES:
    genre_path = os.path.join(DATA_PATH, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
    
    print(f"\n   🖼️  {genre.upper()}:")
    
    for file in tqdm(files[:50], desc=f"   Criando specs {genre}", ncols=70):  # Limitar a 50 por gênero
        audio_path = os.path.join(genre_path, file)
        output_file = file.replace('.wav', '.png')
        output_path = os.path.join(SPECTROGRAM_PATH, genre, output_file)
        
        if create_spectrogram(audio_path, output_path):
            spec_count += 1

print(f"\n   ✅ Espectrogramas criados: {spec_count}")

# ============================================================================
# PREPROCESSAR DADOS
# ============================================================================

print("\n[4/5] Preprocessando dados...")

# Encoder de labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n   Classes encodadas:")
for i, genre in enumerate(le.classes_):
    print(f"      {i} → {genre}")

# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n   ✅ Features normalizadas")
print(f"      Média: {X_scaled.mean():.4f}")
print(f"      Std: {X_scaled.std():.4f}")

# ============================================================================
# SPLIT TRAIN/VALIDATION
# ============================================================================

print("\n[5/5] Dividindo em treino/validação...")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"\n   Treino:     {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Validação:  {X_val.shape[0]} amostras ({X_val.shape[0]/len(X)*100:.1f}%)")

# Verificar distribuição
print(f"\n   Distribuição por classe (treino):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    genre_name = le.classes_[label]
    print(f"      {genre_name:12} {count:3} amostras")

# ============================================================================
# SALVAR DADOS PROCESSADOS
# ============================================================================

print("\n📁 Salvando dados processados...")

# Features
np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_PATH, 'X_val.npy'), X_val)
np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_PATH, 'y_val.npy'), y_val)

# Metadados
np.save(os.path.join(OUTPUT_PATH, 'filenames.npy'), all_filenames)

# Encoders
joblib.dump(le, os.path.join(OUTPUT_PATH, 'label_encoder.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_PATH, 'scaler.pkl'))

# Informações
info = {
    'n_features': X.shape[1],
    'n_classes': len(GENRES),
    'genres': GENRES,
    'sample_rate': SAMPLE_RATE,
    'duration': DURATION,
    'n_mfcc': N_MFCC,
    'n_chroma': N_CHROMA,
    'total_samples': len(X),
    'train_samples': len(X_train),
    'val_samples': len(X_val)
}

import json
with open(os.path.join(OUTPUT_PATH, 'dataset_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print("   ✅ X_train.npy")
print("   ✅ X_val.npy")
print("   ✅ y_train.npy")
print("   ✅ y_val.npy")
print("   ✅ label_encoder.pkl")
print("   ✅ scaler.pkl")
print("   ✅ dataset_info.json")

# ============================================================================
# RESUMO
# ============================================================================

print("\n" + "=" * 80)
print("🎉 PREPARAÇÃO CONCLUÍDA!")
print("=" * 80)

print(f"\n📊 ESTATÍSTICAS:")
print(f"   Total de arquivos: {len(all_features)}")
print(f"   Features por arquivo: {X.shape[1]}")
print(f"   Gêneros: {len(GENRES)}")
print(f"   Espectrogramas: {spec_count}")
print(f"   Train/Val split: {len(X_train)}/{len(X_val)}")

print(f"\n📝 PRÓXIMO PASSO:")
print(f"   python src/treinamento.py")

print("\n" + "=" * 80)