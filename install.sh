#!/bin/bash

# Installation Unifiée pour LLaMA-3-8B Cybersécurité
# Architecture hybride local/cloud avec Google Colab
# Support: Ubuntu/Debian, CentOS/RHEL, macOS
# Version 3.0 - 2025

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Variables de configuration
PROJECT_NAME="LLaMA-CyberSec"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MIN_VERSION="3.8"
PYTHON_RECOMMENDED_VERSION="3.10"
REQUIRED_MEMORY_GB=8
REQUIRED_DISK_GB=15
VENV_NAME="llama_cybersec"
LOGS_DIR="${PROJECT_DIR}/logs"

# Variables globales pour détection
OS=""
PACKAGE_MANAGER=""
GPU_AVAILABLE=false
CUDA_AVAILABLE=false
COLAB_ENV=false
HARDWARE_SPECS=()

# Détecter l'environnement
detect_environment() {
    # Détecter Google Colab
    if [[ -n "${COLAB_GPU}" ]] || [[ -d "/content" ]] || python3 -c "import google.colab" 2>/dev/null; then
        COLAB_ENV=true
        OS="colab"
        log_info "Environnement Google Colab détecté"
        return
    fi
    
    # Détecter le système d'exploitation
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            OS="ubuntu"
            PACKAGE_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            OS="centos"
            PACKAGE_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            OS="fedora"
            PACKAGE_MANAGER="dnf"
        else
            OS="linux"
            PACKAGE_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PACKAGE_MANAGER="brew"
    else
        OS="unknown"
        PACKAGE_MANAGER="unknown"
    fi
    
    log_info "Système détecté: $OS"
}

# Fonctions d'affichage
print_header() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              🛡️  LLaMA-3-8B CYBERSÉCURITÉ 🛡️              ║"
    echo "║                Installation Unifiée v3.0                    ║"
    echo "║              Local + Cloud + Google Colab                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Analyse du hardware
analyze_hardware() {
    log_step "Analyse du hardware système..."
    
    # Détection de l'architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" && "$ARCH" != "arm64" ]]; then
        log_error "Architecture non supportée: $ARCH"
        exit 1
    fi
    
    # Vérification de la RAM
    if [[ "$OS" == "macos" ]]; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    log_info "RAM détectée: ${MEMORY_GB}GB"
    
    if [ "$MEMORY_GB" -lt "$REQUIRED_MEMORY_GB" ]; then
        log_warning "RAM faible (${MEMORY_GB}GB < ${REQUIRED_MEMORY_GB}GB recommandé)"
        if [[ "$COLAB_ENV" == false ]]; then
            read -p "Continuer quand même? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        log_success "RAM suffisante: ${MEMORY_GB}GB"
    fi
    
    # Vérification de l'espace disque
    if [[ "$COLAB_ENV" == false ]]; then
        AVAILABLE_GB=$(df -BG . | awk 'NR==2{print int($4)}')
        log_info "Espace disque disponible: ${AVAILABLE_GB}GB"
        
        if [ "$AVAILABLE_GB" -lt "$REQUIRED_DISK_GB" ]; then
            log_error "Espace disque insuffisant (${AVAILABLE_GB}GB < ${REQUIRED_DISK_GB}GB requis)"
            exit 1
        else
            log_success "Espace disque suffisant"
        fi
    fi
    
    # Détection GPU/CUDA
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        log_success "GPU NVIDIA détecté: $GPU_INFO"
        GPU_AVAILABLE=true
        CUDA_AVAILABLE=true
        HARDWARE_SPECS+=("gpu_nvidia")
    elif command -v rocm-smi &> /dev/null; then
        log_success "GPU AMD ROCm détecté"
        GPU_AVAILABLE=true
        HARDWARE_SPECS+=("gpu_amd")
    elif [[ "$COLAB_ENV" == true ]]; then
        # Vérification spéciale pour Colab
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "GPU Colab")
            log_success "GPU Colab détecté: $GPU_INFO"
            GPU_AVAILABLE=true
            CUDA_AVAILABLE=true
            HARDWARE_SPECS+=("gpu_colab")
        fi
    else
        log_info "Aucun GPU détecté - utilisation CPU uniquement"
        HARDWARE_SPECS+=("cpu_only")
    fi
    
    # Créer les répertoires de logs
    mkdir -p "$LOGS_DIR"
    log_success "Analyse hardware terminée"
}

# Installation des dépendances système
install_system_dependencies() {
    log_step "Installation des dépendances système..."
    
    if [[ "$COLAB_ENV" == true ]]; then
        log_info "Environnement Colab - dépendances déjà présentes"
        return
    fi
    
    case $OS in
        ubuntu)
            sudo apt update && sudo apt upgrade -y
            sudo apt install -y \
                curl wget git unzip \
                build-essential \
                software-properties-common \
                apt-transport-https \
                ca-certificates \
                libssl-dev libffi-dev \
                htop tree vim tmux \
                python3-dev python3-venv
            ;;
        centos)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                curl wget git unzip \
                openssl-devel libffi-devel \
                htop tree vim tmux \
                python3-devel
            ;;
        fedora)
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y \
                curl wget git unzip \
                openssl-devel libffi-devel \
                htop tree vim tmux \
                python3-devel
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install curl wget git unzip htop tree vim tmux
            else
                log_error "Homebrew requis sur macOS. Installez-le depuis https://brew.sh/"
                exit 1
            fi
            ;;
        *)
            log_error "Système non supporté pour l'installation automatique des dépendances"
            exit 1
            ;;
    esac
    
    log_success "Dépendances système installées"
}

# Installation de Python optimisée
install_python() {
    log_step "Configuration de Python ${PYTHON_RECOMMENDED_VERSION}..."
    
    if [[ "$COLAB_ENV" == true ]]; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        log_success "Python Colab: $PYTHON_VERSION"
        return
    fi
    
    # Vérifier la version actuelle
    if command -v python3 &> /dev/null; then
        CURRENT_VERSION=$(python3 --version | awk '{print $2}')
        log_info "Python actuel: $CURRENT_VERSION"
        
        # Vérifier si la version est suffisante
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Version Python OK"
            return
        fi
    fi
    
    case $OS in
        ubuntu)
            # Ajouter le PPA pour les versions récentes
            sudo add-apt-repository ppa:deadsnakes/ppa -y
            sudo apt update
            
            sudo apt install -y \
                python${PYTHON_RECOMMENDED_VERSION} \
                python${PYTHON_RECOMMENDED_VERSION}-dev \
                python${PYTHON_RECOMMENDED_VERSION}-venv \
                python${PYTHON_RECOMMENDED_VERSION}-distutils
            
            # Installer pip
            curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python${PYTHON_RECOMMENDED_VERSION}
            
            # Créer un lien symbolique
            sudo ln -sf /usr/bin/python${PYTHON_RECOMMENDED_VERSION} /usr/local/bin/python3
            ;;
        centos|fedora)
            sudo $PACKAGE_MANAGER install -y python3 python3-pip python3-devel
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install python@${PYTHON_RECOMMENDED_VERSION}
            fi
            ;;
    esac
    
    log_success "Python ${PYTHON_RECOMMENDED_VERSION} installé"
}

# Installation CUDA pour entraînement GPU
install_cuda() {
    if [[ "$CUDA_AVAILABLE" == false ]] || [[ "$COLAB_ENV" == true ]]; then
        log_info "Installation CUDA ignorée"
        return
    fi
    
    log_step "Installation de CUDA (optionnelle)..."
    
    # Vérifier si CUDA est déjà installé
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_success "CUDA déjà installé: $CUDA_VERSION"
        return
    fi
    
    case $OS in
        ubuntu)
            # Installation CUDA pour Ubuntu
            CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
            
            wget -O /tmp/cuda-keyring.deb "$CUDA_KEYRING_URL"
            sudo dpkg -i /tmp/cuda-keyring.deb
            sudo apt update
            
            # Installation toolkit CUDA
            sudo apt install -y cuda-toolkit-11-8
            
            # Ajouter au PATH
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            
            log_success "CUDA installé - redémarrage recommandé"
            ;;
        *)
            log_warning "Installation CUDA automatique non supportée sur $OS"
            log_info "Installez CUDA manuellement depuis https://developer.nvidia.com/cuda-downloads"
            ;;
    esac
}

# Création de l'environnement virtuel
create_virtual_environment() {
    log_step "Création de l'environnement virtuel Python..."
    
    cd "$PROJECT_DIR"
    
    # Déterminer le chemin Python
    if [[ "$COLAB_ENV" == true ]]; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python3"
    fi
    
    # Créer l'environnement virtuel
    if [[ ! -d "$VENV_NAME" ]]; then
        $PYTHON_CMD -m venv "$VENV_NAME"
        log_success "Environnement virtuel créé: $VENV_NAME"
    else
        log_info "Environnement virtuel existant trouvé"
    fi
    
    # Activer l'environnement
    source "${VENV_NAME}/bin/activate"
    
    # Mettre à jour pip
    python -m pip install --upgrade pip wheel setuptools
    
    log_success "Environnement virtuel configuré"
}

# Installation des dépendances Python optimisées
install_python_dependencies() {
    log_step "Installation des dépendances Python optimisées..."
    
    # Activer l'environnement virtuel
    if [[ "$COLAB_ENV" == false ]]; then
        source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    fi
    
    # Installer PyTorch selon la configuration
    if [[ "$GPU_AVAILABLE" == true ]]; then
        log_info "Installation de PyTorch avec support GPU..."
        if [[ "$COLAB_ENV" == true ]]; then
            # Colab a déjà PyTorch, mais on s'assure de la version
            pip install torch torchvision torchaudio --upgrade
        else
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        log_info "Installation de PyTorch CPU uniquement..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Installer les dépendances ML principales
    pip install transformers>=4.35.0
    pip install datasets>=2.14.0
    pip install peft>=0.6.0
    pip install accelerate>=0.24.0
    pip install bitsandbytes>=0.41.0
    
    # Dépendances pour la gestion de données
    pip install pandas>=1.5.0
    pip install numpy>=1.24.0
    pip install scipy>=1.10.0
    
    # Outils de monitoring et logging
    pip install wandb>=0.15.0
    pip install tensorboard>=2.13.0
    pip install tqdm>=4.65.0
    
    # Outils de développement
    pip install jupyter>=1.0.0
    pip install ipython>=8.12.0
    
    # Gestion JSON et YAML
    pip install pyyaml>=6.0
    pip install jsonlines>=3.1.0
    
    # Outils système
    pip install psutil>=5.9.0
    pip install GitPython>=3.1.0
    
    # Web scraping pour enrichissement dataset
    pip install requests>=2.30.0
    pip install beautifulsoup4>=4.12.0
    pip install lxml>=4.9.0
    
    # Installer depuis requirements.txt si disponible
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    log_success "Dépendances Python installées"
}

# Configuration spéciale Google Colab
setup_google_colab() {
    if [[ "$COLAB_ENV" == false ]]; then
        return
    fi
    
    log_step "Configuration Google Colab..."
    
    # Monter Google Drive
    python3 -c "
from google.colab import drive
drive.mount('/content/drive')
print('Google Drive monté avec succès')
" 2>/dev/null || log_warning "Impossible de monter Google Drive"
    
    # Créer les répertoires sur Drive
    DRIVE_PROJECT_DIR="/content/drive/MyDrive/LLaMA_CyberSec"
    mkdir -p "$DRIVE_PROJECT_DIR"/{models,datasets,logs,configs}
    
    # Créer des liens symboliques vers Drive pour persistance
    if [[ -d "/content/drive/MyDrive" ]]; then
        ln -sf "$DRIVE_PROJECT_DIR/models" "$PROJECT_DIR/models_persistent"
        ln -sf "$DRIVE_PROJECT_DIR/logs" "$PROJECT_DIR/logs_persistent"
        log_success "Persistance Google Drive configurée"
    fi
    
    # Installer les outils Colab spécifiques
    pip install google-colab
    
    log_success "Configuration Colab terminée"
}

# Création des scripts de lancement
create_launch_scripts() {
    log_step "Création des scripts de lancement..."
    
    # Script d'activation de l'environnement
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Script d'activation de l'environnement LLaMA CyberSec

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Détecter l'environnement
if [[ -n "${COLAB_GPU}" ]] || [[ -d "/content" ]]; then
    echo "🌐 Environnement Google Colab détecté"
    PYTHON_CMD="python3"
else
    echo "🖥️  Environnement local détecté"
    source "${PROJECT_DIR}/llama_cybersec/bin/activate"
    PYTHON_CMD="python"
fi

echo "✅ Environnement activé"
echo "📁 Projet: $PROJECT_DIR"
echo "🐍 Python: $($PYTHON_CMD --version)"

# Vérifier PyTorch
if $PYTHON_CMD -c "import torch; print('🔧 PyTorch:', torch.__version__)" 2>/dev/null; then
    if $PYTHON_CMD -c "import torch; print('🚀 CUDA disponible:', torch.cuda.is_available())" | grep -q "True"; then
        $PYTHON_CMD -c "import torch; print('🎮 GPU:', torch.cuda.get_device_name(0))"
    fi
else
    echo "⚠️  PyTorch non détecté"
fi

echo ""
echo "📋 Commandes disponibles:"
echo "  python scripts/dataset_manager.py --help"
echo "  python scripts/train_lora.py --help"
echo "  python scripts/merge_and_quantize.py --help"
echo "  python cloud_training/cloud_trainer.py --help"
EOF
    
    chmod +x activate_env.sh
    
    # Script de démarrage rapide
    cat > quick_start.sh << 'EOF'
#!/bin/bash
# Démarrage rapide du système LLaMA Cybersécurité

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Démarrage rapide du système LLaMA Cybersécurité"

# Activer l'environnement
source activate_env.sh

echo ""
echo "📊 Vérification du système..."

# Vérifier les composants essentiels
python -c "
import sys
try:
    import torch
    import transformers
    import datasets
    import peft
    print('✅ Tous les composants ML sont présents')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   Transformers: {transformers.__version__}')
    print(f'   PEFT: {peft.__version__}')
except ImportError as e:
    print(f'❌ Composant manquant: {e}')
    sys.exit(1)
"

echo ""
echo "🎯 Prochaines étapes:"
echo "1. 📝 Créer un dataset:"
echo "   python scripts/dataset_manager.py --create-initial"
echo ""
echo "2. 🚀 Lancer un entraînement:"
echo "   python scripts/train_lora.py"
echo ""
echo "3. ☁️  Utiliser Google Colab:"
echo "   python cloud_training/cloud_trainer.py"
echo ""
echo "✨ Le système est prêt à l'emploi!"
EOF
    
    chmod +x quick_start.sh
    
    # Script pour Google Colab
    cat > colab_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Script de configuration automatique pour Google Colab
Usage: python colab_setup.py
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    print("🌐 Configuration automatique pour Google Colab")
    
    # Vérifier qu'on est bien sur Colab
    try:
        import google.colab
        print("✅ Environnement Google Colab confirmé")
    except ImportError:
        print("❌ Ce script est conçu pour Google Colab uniquement")
        sys.exit(1)
    
    # Monter Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive monté")
    except Exception as e:
        print(f"⚠️  Erreur montage Drive: {e}")
    
    # Installer les dépendances manquantes
    packages = [
        "transformers>=4.35.0",
        "datasets>=2.14.0", 
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "wandb>=0.15.0"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                         check=True)
            print(f"✅ {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"❌ Erreur installation {package}")
    
    # Configurer les répertoires persistants
    drive_path = Path("/content/drive/MyDrive/LLaMA_CyberSec")
    drive_path.mkdir(exist_ok=True)
    
    for subdir in ["models", "datasets", "logs", "checkpoints"]:
        (drive_path / subdir).mkdir(exist_ok=True)
    
    print("✅ Répertoires persistants créés sur Drive")
    
    # Test rapide
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  GPU non disponible")
    except ImportError:
        print("❌ PyTorch non disponible")
    
    print("\n🎉 Configuration Colab terminée!")
    print("📋 Prochaines étapes:")
    print("1. Créer un dataset: !python scripts/dataset_manager.py --create-initial")
    print("2. Entraîner: !python cloud_training/cloud_trainer.py")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x colab_setup.py
    
    log_success "Scripts de lancement créés"
}

# Tests de validation du système
run_system_tests() {
    log_step "Exécution des tests de validation..."
    
    # Activer l'environnement
    if [[ "$COLAB_ENV" == false ]]; then
        source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    fi
    
    # Test d'importation des modules principaux
    python3 -c "
import sys
try:
    import torch
    import transformers
    import datasets
    import peft
    import bitsandbytes
    print('✅ Tous les modules ML importés avec succès')
    
    # Test PyTorch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA disponible: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    print('✅ Tests de validation réussis')
    
except ImportError as e:
    print(f'❌ Erreur d\\'importation: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Erreur: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Tests système réussis"
    else
        log_error "Tests système échoués"
        exit 1
    fi
}

# Affichage du résumé final
show_completion_summary() {
    echo
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 INSTALLATION TERMINÉE AVEC SUCCÈS! 🎉${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}📁 Répertoire du projet:${NC} $PROJECT_DIR"
    
    if [[ "$COLAB_ENV" == true ]]; then
        echo -e "${CYAN}🌐 Environnement:${NC} Google Colab"
        echo -e "${CYAN}💾 Stockage persistant:${NC} Google Drive"
    else
        echo -e "${CYAN}🐍 Environnement virtuel:${NC} $VENV_NAME"
        echo -e "${CYAN}📊 Logs:${NC} $LOGS_DIR"
    fi
    
    # Afficher les spécifications hardware
    echo -e "${CYAN}🖥️  Hardware:${NC}"
    echo "   • RAM: ${MEMORY_GB}GB"
    if [[ "$GPU_AVAILABLE" == true ]]; then
        echo "   • GPU: ✅ Disponible"
    else
        echo "   • GPU: ❌ Non disponible (CPU uniquement)"
    fi
    
    echo -e "${YELLOW}🚀 DÉMARRAGE RAPIDE:${NC}"
    
    if [[ "$COLAB_ENV" == true ]]; then
        echo "1. Configuration Colab:"
        echo "   !python colab_setup.py"
        echo
        echo "2. Créer un dataset:"
        echo "   !python scripts/dataset_manager.py --create-initial"
        echo
        echo "3. Lancer l'entraînement:"
        echo "   !python cloud_training/cloud_trainer.py"
    else
        echo "1. Activer l'environnement:"
        echo "   source activate_env.sh"
        echo
        echo "2. Configuration rapide:"
        echo "   ./quick_start.sh"
        echo
        echo "3. Créer un dataset:"
        echo "   python scripts/dataset_manager.py --create-initial"
        echo
        echo "4. Entraîner localement:"
        echo "   python scripts/train_lora.py"
        echo
        echo "5. Entraîner sur le cloud:"
        echo "   python cloud_training/cloud_trainer.py"
    fi
    
    echo -e "${YELLOW}📚 DOCUMENTATION:${NC}"
    echo "• README.md: Guide complet"
    echo "• docs/: Documentation détaillée" 
    echo "• ANALYSE_PROJET_ET_PLAN.md: Plan d'amélioration"
    
    echo -e "${YELLOW}💡 CONSEILS:${NC}"
    if [[ "$COLAB_ENV" == true ]]; then
        echo "• Utilisez les GPU gratuits de Colab pour l'entraînement"
        echo "• Sauvegardez régulièrement sur Google Drive"
        echo "• Surveillez les quotas Colab"
    else
        echo "• Utilisez tmux pour les longs entraînements"
        echo "• Surveillez l'utilisation GPU avec nvidia-smi"
        echo "• Consultez les logs dans $LOGS_DIR"
    fi
    
    echo
    echo -e "${GREEN}✨ Votre système LLaMA Cybersécurité est prêt! ✨${NC}"
}

# Configuration interactive optionnelle
interactive_setup() {
    if [[ "$COLAB_ENV" == true ]]; then
        return  # Pas de configuration interactive sur Colab
    fi
    
    echo
    echo -e "${CYAN}🔧 CONFIGURATION INTERACTIVE${NC}"
    echo "Voulez-vous configurer le système maintenant ?"
    echo
    echo "1. Configuration complète (recommandée)"
    echo "2. Configuration cloud uniquement" 
    echo "3. Configuration locale uniquement"
    echo "4. Passer pour l'instant"
    echo
    
    read -p "Votre choix (1-4): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            log_info "Configuration complète sélectionnée"
            ./quick_start.sh
            ;;
        2)
            log_info "Configuration cloud sélectionnée"
            python cloud_training/cloud_trainer.py --help
            ;;
        3)
            log_info "Configuration locale sélectionnée"
            python scripts/dataset_manager.py --create-initial
            ;;
        4)
            log_info "Configuration reportée"
            ;;
        *)
            log_info "Choix invalide - configuration reportée"
            ;;
    esac
}

# Fonction principale
main() {
    print_header
    
    log_info "Début de l'installation unifiée..."
    
    # Phase 1: Analyse de l'environnement
    detect_environment
    analyze_hardware
    
    # Phase 2: Installation des dépendances
    install_system_dependencies
    install_python
    
    # Phase 3: Configuration GPU/CUDA
    if [[ "$GPU_AVAILABLE" == true && "$COLAB_ENV" == false ]]; then
        install_cuda
    fi
    
    # Phase 4: Environnement Python
    create_virtual_environment
    install_python_dependencies
    
    # Phase 5: Configuration spécialisée
    setup_google_colab
    create_launch_scripts
    
    # Phase 6: Tests et validation
    run_system_tests
    
    # Phase 7: Configuration interactive
    interactive_setup
    
    # Phase 8: Résumé final
    show_completion_summary
    
    log_success "Installation unifiée terminée avec succès!"
}

# Gestion des arguments
case "${1:-}" in
    --help|-h)
        echo "Installation Unifiée LLaMA-3-8B Cybersécurité"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h         Afficher cette aide"
        echo "  --no-gpu          Forcer installation CPU uniquement"
        echo "  --no-cuda         Ignorer l'installation CUDA"
        echo "  --colab           Forcer mode Google Colab"
        echo "  --quick           Installation rapide automatique"
        echo "  --interactive     Mode interactif complet"
        echo ""
        echo "Exemples:"
        echo "  $0                    # Installation automatique"
        echo "  $0 --quick           # Installation rapide"
        echo "  $0 --no-gpu          # Sans support GPU"
        echo "  $0 --colab           # Mode Google Colab"
        echo ""
        exit 0
        ;;
    --no-gpu)
        GPU_AVAILABLE=false
        CUDA_AVAILABLE=false
        main
        ;;
    --no-cuda)
        CUDA_AVAILABLE=false
        main
        ;;
    --colab)
        COLAB_ENV=true
        main
        ;;
    --quick)
        INTERACTIVE=false
        main
        ;;
    --interactive)
        INTERACTIVE=true
        main
        ;;
    *)
        main
        ;;
esac