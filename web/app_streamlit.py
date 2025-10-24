"""
app_streamlit.py
Interface web para classificação de gêneros musicais
"""

import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import json
from tensorflow import keras
import tempfile

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .genre-name {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE DOS MODELOS
# ============================================================================

@st.cache_resource
def load_model():
    """Carrega modelo e preprocessadores"""
    try:
        modelo = keras.models.load_model('models/music_genre_classifier.h5')
        scaler = joblib.load('data/processed/scaler.pkl')
        label_encoder = joblib.load('data/processed/label_encoder.pkl')
        
        with open('data/processed/dataset_info.json', 'r') as f:
            info = json.load(f)
        
        return modelo, scaler, label_encoder, info
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        st.info("Execute primeiro: python src/02_train_model.py")
        return None, None, None, None

# ============================================================================
# FUNÇÕES
# ============================================================================

def extract_features(audio_path, n_mfcc=13, n_chroma=12):
    """Extrai features de áudio"""
    y, sr = librosa.load(audio_path, sr=22050, duration=30)
    
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
    
    return features, y, sr

def plot_waveform(y, sr):
    """Plota forma de onda"""
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.8, ax=ax, color='#667eea')
    ax.set_title('🎵 Waveform', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    return fig

def plot_spectrogram(y, sr):
    """Plota espectrograma"""
    fig, ax = plt.subplots(figsize=(12, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', 
                                   ax=ax, cmap='viridis')
    ax.set_title('📊 Mel Spectrogram', fontsize=14, fontweight='bold', pad=15)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    return fig

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

# Header
st.markdown('<p class="main-header">🎵 Music Genre Classifier</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Descubra o gênero musical usando Inteligência Artificial</p>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("ℹ️ Sobre")
    st.info("""
    Este app usa **Deep Learning** para classificar músicas em 10 gêneros diferentes.
    
    **Gêneros suportados:**
    - 🎸 Blues
    - 🎻 Classical
    - 🤠 Country
    - 🕺 Disco
    - 🎤 Hip Hop
    - 🎷 Jazz
    - 🤘 Metal
    - 🎤 Pop
    - 🇯🇲 Reggae
    - 🎸 Rock
    """)
    
    st.title("🎯 Como usar")
    st.markdown("""
    1. 📁 Faça upload de um arquivo de áudio
    2. ⏳ Aguarde o processamento
    3. 🎉 Veja o resultado!
    """)
    
    st.title("📊 Seu Modelo")
    st.success("""
    **Acurácia: 91.5%** 🏆
    
    Confiança média: 85.95%
    
    183 acertos / 17 erros
    """)

# Carregar modelo
modelo, scaler, label_encoder, info = load_model()

if modelo is None:
    st.stop()

genres = info['genres']
n_mfcc = info['n_mfcc']
n_chroma = info['n_chroma']

# Mapeamento de emojis
emoji_map = {
    'blues': '🎸', 'classical': '🎻', 'country': '🤠',
    'disco': '🕺', 'hiphop': '🎤', 'jazz': '🎷',
    'metal': '🤘', 'pop': '🎤', 'reggae': '🇯🇲', 'rock': '🎸'
}

# Upload de arquivo
st.markdown("### 📁 Upload de Arquivo")

uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio (MP3, WAV, OGG)",
    type=['mp3', 'wav', 'ogg'],
    help="Arraste e solte ou clique para selecionar"
)

if uploaded_file is not None:
    
    # Salvar temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name
    
    # Informações do arquivo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Arquivo", uploaded_file.name)
    with col2:
        st.metric("📦 Tamanho", f"{uploaded_file.size / 1024:.1f} KB")
    with col3:
        try:
            y_temp, sr_temp = librosa.load(audio_path, sr=None)
            duracao = len(y_temp) / sr_temp
            st.metric("⏱️ Duração", f"{duracao:.1f}s")
        except:
            st.metric("⏱️ Duração", "N/A")
    
    st.markdown("---")
    
    # Player de áudio
    st.markdown("### 🎧 Player")
    st.audio(uploaded_file, format='audio/mp3')
    
    st.markdown("---")
    
    # Processar
    with st.spinner('🎵 Analisando áudio...'):
        try:
            # Extrair features
            features, y, sr = extract_features(audio_path, n_mfcc, n_chroma)
            
            # Normalizar
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Predição
            predictions = modelo.predict(features_scaled, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
            confidence = predictions[predicted_class]
            
            # Resultado principal
            st.markdown("### 🏆 Resultado")
            
            emoji = emoji_map.get(predicted_genre, '🎵')
            
            st.markdown(f"""
            <div class="result-box">
                <div class="genre-name">{emoji} {predicted_genre.upper()}</div>
                <div class="confidence">Confiança: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # TOP 3
            st.markdown("### 📊 TOP 3 Predições")
            
            top3_indices = np.argsort(predictions)[::-1][:3]
            cols = st.columns(3)
            
            medals = ['🥇', '🥈', '🥉']
            
            for i, (col, idx) in enumerate(zip(cols, top3_indices)):
                genre = label_encoder.inverse_transform([idx])[0]
                prob = predictions[idx]
                emoji_g = emoji_map.get(genre, '🎵')
                
                with col:
                    st.metric(
                        label=f"{medals[i]} {emoji_g} {genre.capitalize()}",
                        value=f"{prob*100:.1f}%"
                    )
            
            # Todas as predições
            st.markdown("### 📈 Todas as Predições")
            
            pred_data = {}
            for i, genre in enumerate(genres):
                emoji_g = emoji_map.get(genre, '🎵')
                pred_data[f"{emoji_g} {genre.capitalize()}"] = predictions[i]
            
            st.bar_chart(pred_data)
            
            # Visualizações
            st.markdown("### 🎨 Visualizações")
            
            tab1, tab2 = st.tabs(["🎵 Waveform", "📊 Spectrogram"])
            
            with tab1:
                fig_wave = plot_waveform(y, sr)
                st.pyplot(fig_wave)
            
            with tab2:
                fig_spec = plot_spectrogram(y, sr)
                st.pyplot(fig_spec)
            
            # Limpar arquivo temporário
            os.unlink(audio_path)
            
        except Exception as e:
            st.error(f"❌ Erro ao processar: {e}")
            st.info("Certifique-se de que o arquivo é válido.")

else:
    # Instruções
    st.info("👆 Faça upload de um arquivo de áudio para começar")
    
    # Exemplos
    st.markdown("### 💡 Dica")
    st.markdown("""
    Para melhores resultados:
    - Use arquivos com pelo menos 30 segundos
    - Certifique-se de que o áudio tem boa qualidade
    - Evite músicas com muito ruído de fundo
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Desenvolvido com ❤️ usando TensorFlow, Librosa e Streamlit</p>
    <p>🎵 Music Genre Classifier © 2025 | Acurácia: 91.5% 🏆</p>
</div>
""", unsafe_allow_html=True)