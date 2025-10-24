"""
Download e preparação automática do dataset GTZAN
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm

print("=" * 80)
print("🎵 DOWNLOAD DO DATASET GTZAN")
print("=" * 80)

# URLs do dataset
DATASET_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
DATASET_PATH = "data/raw"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

print("\n📊 OPÇÕES DE DOWNLOAD:")
print("\n1️⃣  Download Automático (Oficial - pode estar lento)")
print("   URL:", DATASET_URL)
print("\n2️⃣  Download Manual do Kaggle (Recomendado)")
print("   URL: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
print("\n3️⃣  Download do Google Drive (Alternativo)")
print("   URL: https://drive.google.com/...")

choice = input("\n❓ Escolha uma opção (1/2/3): ").strip()

if choice == "1":
    print("\n📥 Iniciando download automático...")
    print("⏳ Isso pode levar alguns minutos (~1.2 GB)")
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    output_file = os.path.join(DATASET_PATH, "genres.tar.gz")
    
    try:
        download_url(DATASET_URL, output_file)
        print("\n✅ Download concluído!")
        
        print("\n📦 Extraindo arquivos...")
        import tarfile
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall(DATASET_PATH)
        
        print("✅ Extração concluída!")
        print(f"\n📁 Dataset salvo em: {DATASET_PATH}/genres/")
        
        # Limpar arquivo zip
        os.remove(output_file)
        print("🧹 Arquivo temporário removido")
        
    except Exception as e:
        print(f"\n❌ Erro no download: {e}")
        print("\n💡 Tente a opção manual (opção 2)")

elif choice == "2":
    print("\n📝 INSTRUÇÕES PARA DOWNLOAD MANUAL:")
    print("\n1. Acesse: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    print("2. Clique em 'Download' (requer conta Kaggle)")
    print("3. Extraia o arquivo 'archive.zip'")
    print("4. Mova a pasta 'Data/genres_original/' para 'data/raw/'")
    print("\n📁 Estrutura final esperada:")
    print("   data/raw/")
    print("   ├── blues/")
    print("   ├── classical/")
    print("   ├── country/")
    print("   ├── disco/")
    print("   ├── hiphop/")
    print("   ├── jazz/")
    print("   ├── metal/")
    print("   ├── pop/")
    print("   ├── reggae/")
    print("   └── rock/")
    
    input("\n⏸️  Pressione ENTER quando terminar...")
    
elif choice == "3":
    print("\n💡 Entre em contato para link do Google Drive")
    print("   Ou use a opção 2 (Kaggle)")

else:
    print("\n❌ Opção inválida!")

# Verificar se dataset foi baixado
print("\n🔍 Verificando dataset...")

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

all_found = True
for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    if os.path.exists(genre_path):
        n_files = len([f for f in os.listdir(genre_path) if f.endswith('.wav')])
        print(f"   ✅ {genre:12} {n_files} arquivos")
    else:
        print(f"   ❌ {genre:12} não encontrado")
        all_found = False

if all_found:
    print("\n" + "=" * 80)
    print("🎉 DATASET PRONTO!")
    print("=" * 80)
    print("\n📝 Próximo passo:")
    print("   python src/01_prepare_data.py")
else:
    print("\n" + "=" * 80)
    print("⚠️  DATASET INCOMPLETO")
    print("=" * 80)
    print("\n💡 Siga as instruções da opção 2 (download manual)")

print()