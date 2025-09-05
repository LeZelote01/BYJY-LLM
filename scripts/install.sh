#!/bin/bash

# Script d'installation automatisée pour le système de fine-tuning LLaMA-3-8B
# Conçu pour Ubuntu 22.04 avec support CUDA optionnel

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables de configuration
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
VENV_NAME="llama_cybersec"
LOGS_DIR="${PROJECT_DIR}/logs"

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_system() {
    log_info "Vérification du système..."
    
    # Vérifier Ubuntu
    if ! grep -q "Ubuntu" /etc/os-release; then
        log_warning "Ce script est optimisé pour Ubuntu 22.04"
    fi
    
    # Vérifier l'architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        log_error "Architecture non supportée: $ARCH"
        exit 1
    fi
    
    # Vérifier la RAM
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $RAM_GB -lt 16 ]]; then
        log_warning "RAM détectée: ${RAM_GB}GB. Recommandé: 20GB minimum"
        read -p "Continuer quand même? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "RAM détectée: ${RAM_GB}GB"
    fi
    
    # Vérifier CUDA
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        log_success "GPU détecté: $GPU_INFO"
        CUDA_AVAILABLE=true
    else
        log_warning "NVIDIA GPU non détecté - entraînement CPU uniquement"
        CUDA_AVAILABLE=false
    fi
    
    # Créer les répertoires
    mkdir -p "$LOGS_DIR"
    log_success "Système vérifié"
}

install_system_dependencies() {
    log_info "Installation des dépendances système..."
    
    # Mise à jour du système
    sudo apt update && sudo apt upgrade -y
    
    # Dépendances de base
    sudo apt install -y \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        unzip \
        htop \
        tree \
        vim \
        tmux
    
    log_success "Dépendances système installées"
}

install_python() {
    log_info "Installation de Python ${PYTHON_VERSION}..."
    
    # Ajouter le PPA pour Python
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    
    # Installer Python et pip
    sudo apt install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils
    
    # Installer pip pour cette version de Python
    curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python${PYTHON_VERSION}
    
    # Créer un lien symbolique
    sudo ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3
    
    log_success "Python ${PYTHON_VERSION} installé"
}

install_cuda() {
    if [[ "$CUDA_AVAILABLE" == false ]]; then
        log_info "Installation de CUDA ignorée (GPU non détecté)"
        return
    fi
    
    log_info "Installation de CUDA ${CUDA_VERSION}..."
    
    # Télécharger et installer CUDA
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
    
    wget -O /tmp/cuda-keyring.deb "$CUDA_KEYRING_URL"
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt update
    
    # Installer CUDA toolkit
    sudo apt install -y cuda-toolkit-${CUDA_VERSION//./-}
    
    # Ajouter CUDA au PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Installer cuDNN (optionnel mais recommandé)
    log_info "Installation de cuDNN..."
    sudo apt install -y libcudnn8 libcudnn8-dev
    
    log_success "CUDA ${CUDA_VERSION} installé"
}

create_virtual_environment() {
    log_info "Création de l'environnement virtuel Python..."
    
    cd "$PROJECT_DIR"
    
    # Créer l'environnement virtuel
    python3 -m venv "$VENV_NAME"
    
    # Activer l'environnement
    source "${VENV_NAME}/bin/activate"
    
    # Mettre à jour pip
    pip install --upgrade pip wheel setuptools
    
    log_success "Environnement virtuel créé: $VENV_NAME"
}

install_python_dependencies() {
    log_info "Installation des dépendances Python..."
    
    cd "$PROJECT_DIR"
    source "${VENV_NAME}/bin/activate"
    
    # Créer le fichier requirements.txt s'il n'existe pas
    if [[ ! -f "requirements.txt" ]]; then
        log_info "Création du fichier requirements.txt..."
        cat > requirements.txt << EOF
# Core ML libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.0
accelerate>=0.20.0

# LoRA and quantization
peft>=0.4.0
bitsandbytes>=0.39.0

# Data processing
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0

# Web scraping and parsing
requests>=2.30.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Git operations
GitPython>=3.1.0

# Logging and monitoring
wandb>=0.15.0
tensorboard>=2.13.0

# System monitoring
psutil>=5.9.0

# JSON and YAML processing
pyyaml>=6.0
jsonlines>=3.1.0

# Progress bars
tqdm>=4.65.0

# Development tools
jupyter>=1.0.0
ipython>=8.12.0

# Optional: for better performance
ninja>=1.11.0
EOF
    fi
    
    # Installer PyTorch avec support CUDA si disponible
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        log_info "Installation de PyTorch avec support CUDA..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installation de PyTorch CPU uniquement..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Installer les autres dépendances
    pip install -r requirements.txt
    
    log_success "Dépendances Python installées"
}

download_base_model() {
    log_info "Configuration pour le téléchargement du modèle LLaMA..."
    
    # Créer le répertoire pour les modèles
    mkdir -p "${PROJECT_DIR}/models/base"
    
    # Instructions pour obtenir l'accès au modèle
    cat << EOF

${YELLOW}=== ACCÈS AU MODÈLE LLAMA ===

Pour télécharger LLaMA-3-8B, vous devez:

1. Créer un compte Hugging Face: https://huggingface.co/
2. Demander l'accès au modèle: https://huggingface.co/meta-llama/Llama-2-7b-hf
3. Installer Hugging Face CLI:
   pip install huggingface_hub
4. Se connecter:
   huggingface-cli login
5. Le modèle sera téléchargé automatiquement lors du premier entraînement

Modèles recommandés:
- meta-llama/Llama-2-7b-hf (7B paramètres)
- meta-llama/Llama-2-13b-hf (13B paramètres, plus de ressources)

${NC}
EOF
    
    read -p "Appuyez sur Entrée pour continuer..."
}

setup_git_repositories() {
    log_info "Configuration des repositories Git..."
    
    cd "$PROJECT_DIR"
    
    # Initialiser le dépôt git s'il n'existe pas
    if [[ ! -d ".git" ]]; then
        git init
        log_info "Repository Git initialisé"
    fi
    
    # Créer .gitignore s'il n'existe pas
    if [[ ! -f ".gitignore" ]]; then
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
${VENV_NAME}/

# Machine Learning
*.pth
*.pt
*.bin
*.safetensors
models/base/
models/quantized/
*.log

# Data
datasets/sources/
*.jsonl.backup

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Weights & Biases
wandb/

# Jupyter
.ipynb_checkpoints/

# Temporary files
*.tmp
*.temp
/tmp/
EOF
        log_success ".gitignore créé"
    fi
}

create_service_scripts() {
    log_info "Création des scripts de service..."
    
    cd "$PROJECT_DIR"
    
    # Script d'activation de l'environnement
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Script d'activation de l'environnement virtuel

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_DIR}/llama_cybersec/bin/activate"

echo "✅ Environnement virtuel activé"
echo "📁 Projet: $PROJECT_DIR"
echo "🐍 Python: $(python --version)"
echo "🔧 PyTorch: $(python -c 'import torch; print(torch.__version__)')"

if python -c 'import torch; print("CUDA disponible:", torch.cuda.is_available())' | grep -q "True"; then
    echo "🚀 CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Non disponible")')"
fi

echo ""
echo "Commandes disponibles:"
echo "  python scripts/dataset_manager.py --help"
echo "  python scripts/train_lora.py --help" 
echo "  python scripts/merge_and_quantize.py --help"
EOF
    
    chmod +x activate_env.sh
    
    # Script de démarrage rapide
    cat > quick_start.sh << 'EOF'
#!/bin/bash
# Script de démarrage rapide

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Démarrage rapide du système LLaMA Cybersécurité"

# Activer l'environnement
source llama_cybersec/bin/activate

# Créer le dataset initial
echo "📝 Création du dataset initial..."
python scripts/dataset_manager.py --create-initial

# Enrichir le dataset (optionnel, peut prendre du temps)
read -p "Enrichir le dataset depuis les sources publiques? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌐 Enrichissement du dataset..."
    python scripts/dataset_manager.py --enrich-all
fi

# Afficher les statistiques
echo "📊 Statistiques du dataset:"
python scripts/dataset_manager.py --stats

echo ""
echo "✅ Configuration terminée!"
echo "📋 Prochaines étapes:"
echo "  1. Vérifiez le dataset: python scripts/dataset_manager.py --list"
echo "  2. Estimez le temps: python scripts/train_lora.py --estimate-time" 
echo "  3. Lancez l'entraînement: python scripts/train_lora.py"
EOF
    
    chmod +x quick_start.sh
    
    log_success "Scripts de service créés"
}

create_config_files() {
    log_info "Création des fichiers de configuration..."
    
    cd "$PROJECT_DIR"
    mkdir -p configs
    
    # Configuration d'entraînement par défaut
    cat > configs/training_config.json << EOF
{
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_path": "/app/datasets/enriched_dataset.jsonl",
    "output_dir": "/app/models/lora_weights",
    
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },
    
    "training_args": {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "logging_steps": 25,
        "save_steps": 500,
        "eval_steps": 500,
        "fp16": true,
        "dataloader_num_workers": 4
    },
    
    "quantization": {
        "load_in_4bit": true,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": true
    }
}
EOF
    
    # Configuration de dataset
    cat > configs/dataset_config.json << EOF
{
    "sources": {
        "sigma_rules": {
            "url": "https://github.com/SigmaHQ/sigma.git",
            "enabled": true,
            "max_examples": 100
        },
        "yara_rules": {
            "url": "https://github.com/Yara-Rules/rules.git", 
            "enabled": true,
            "max_examples": 50
        },
        "atomic_red_team": {
            "url": "https://github.com/redcanaryco/atomic-red-team.git",
            "enabled": true,
            "max_examples": 75
        },
        "cybersec_scripts": {
            "url": "https://github.com/danielmiessler/SecLists.git",
            "enabled": false,
            "max_examples": 25
        }
    },
    
    "processing": {
        "max_length": 2048,
        "min_instruction_length": 10,
        "min_output_length": 20,
        "deduplicate": true,
        "shuffle": true
    }
}
EOF
    
    log_success "Fichiers de configuration créés"
}

run_system_tests() {
    log_info "Exécution des tests système..."
    
    cd "$PROJECT_DIR"
    source "${VENV_NAME}/bin/activate"
    
    # Test d'importation des bibliothèques principales
    echo "🧪 Test des imports Python..."
    python -c "
import torch
import transformers
import peft
import bitsandbytes
import datasets
print('✅ Toutes les bibliothèques sont importables')
"
    
    # Test PyTorch
    echo "🧪 Test PyTorch..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
    
    # Test des scripts
    echo "🧪 Test des scripts..."
    python scripts/dataset_manager.py --help > /dev/null
    python scripts/train_lora.py --help > /dev/null
    
    log_success "Tests système réussis"
}

show_completion_message() {
    cat << EOF

${GREEN}
🎉 INSTALLATION TERMINÉE AVEC SUCCÈS! 🎉

Votre système de fine-tuning LLaMA-3-8B pour la cybersécurité est prêt!

📁 Répertoire du projet: $PROJECT_DIR
🐍 Environnement virtuel: $VENV_NAME
📊 Logs: $LOGS_DIR

🚀 DÉMARRAGE RAPIDE:
1. Activer l'environnement:
   source activate_env.sh

2. Configuration initiale:
   ./quick_start.sh

3. Créer un dataset:
   python scripts/dataset_manager.py --create-initial

4. Enrichir le dataset:
   python scripts/dataset_manager.py --enrich-all

5. Lancer l'entraînement:
   python scripts/train_lora.py

📚 DOCUMENTATION:
- README.md: Guide complet
- docs/: Documentation détaillée
- configs/: Fichiers de configuration

💡 CONSEILS:
- Utilisez tmux pour les longs entraînements
- Surveillez l'utilisation GPU avec nvidia-smi
- Consultez les logs dans ${LOGS_DIR}

${NC}
EOF
}

# Fonction principale
main() {
    echo "🚀 Installation du système LLaMA-3-8B Cybersécurité"
    echo "=================================================="
    
    check_system
    install_system_dependencies
    install_python
    
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        install_cuda
    fi
    
    create_virtual_environment
    install_python_dependencies
    download_base_model
    setup_git_repositories
    create_service_scripts
    create_config_files
    run_system_tests
    
    show_completion_message
}

# Gestion des arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script exécuté directement
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --help, -h     Afficher cette aide"
            echo "  --no-cuda      Ignorer l'installation CUDA"
            echo "  --python-only  Installer seulement Python et les dépendances"
            exit 0
            ;;
        --no-cuda)
            CUDA_AVAILABLE=false
            main
            ;;
        --python-only)
            create_virtual_environment
            install_python_dependencies 
            ;;
        *)
            main
            ;;
    esac
fi