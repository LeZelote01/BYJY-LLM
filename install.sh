#!/bin/bash

# Installation automatique pour LLaMA-3-8B Cybersécurité
# Support: Ubuntu/Debian, CentOS/RHEL, macOS
# Version 2.0 avec architecture cloud/local

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
PYTHON_MIN_VERSION="3.8"
REQUIRED_MEMORY_GB=8
REQUIRED_DISK_GB=10

# Détecter le système
detect_system() {
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
}

# Fonctions d'affichage
print_header() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                  🛡️  LLaMA-3-8B CYBERSÉCURITÉ                ║"
    echo "║                    Installation Automatique                  ║"
    echo "║                      Version 2.0 - 2025                     ║"
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

# Vérification des prérequis système
check_system_requirements() {
    log_step "Vérification des prérequis système..."
    
    # Vérifier l'OS
    detect_system
    log_info "Système détecté: $OS"
    
    # Vérifier Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        log_info "Python détecté: $PYTHON_VERSION"
        
        # Vérifier la version minimale
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Version Python OK"
        else
            log_error "Python 3.8+ requis. Version actuelle: $PYTHON_VERSION"
            install_python
        fi
    else
        log_warning "Python 3 non détecté"
        install_python
    fi
    
    # Vérifier pip
    if command -v pip3 &> /dev/null; then
        log_success "pip3 détecté"
    else
        log_warning "pip3 non détecté, installation..."
        install_pip
    fi
    
    # Vérifier git
    if command -v git &> /dev/null; then
        log_success "Git détecté"
    else
        log_warning "Git non détecté, installation..."
        install_git
    fi
    
    # Vérifier la mémoire
    check_memory
    
    # Vérifier l'espace disque
    check_disk_space
    
    # Vérifier GPU (optionnel)
    check_gpu
}

install_python() {
    log_step "Installation de Python 3..."
    
    case $OS in
        ubuntu)
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv python3-dev
            ;;
        centos)
            sudo yum install -y python3 python3-pip python3-devel
            ;;
        fedora)
            sudo dnf install -y python3 python3-pip python3-devel
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install python@3.10
            else
                log_error "Homebrew requis sur macOS. Installez-le depuis https://brew.sh/"
                exit 1
            fi
            ;;
        *)
            log_error "Système non supporté pour l'installation automatique de Python"
            log_info "Veuillez installer Python 3.8+ manuellement"
            exit 1
            ;;
    esac
    
    log_success "Python installé"
}

install_pip() {
    log_step "Installation de pip..."
    
    if command -v python3 &> /dev/null; then
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    else
        log_error "Python 3 requis pour installer pip"
        exit 1
    fi
    
    log_success "pip installé"
}

install_git() {
    log_step "Installation de Git..."
    
    case $OS in
        ubuntu)
            sudo apt update
            sudo apt install -y git
            ;;
        centos)
            sudo yum install -y git
            ;;
        fedora)
            sudo dnf install -y git
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install git
            else
                log_info "Git peut être installé avec Xcode Command Line Tools"
                xcode-select --install
            fi
            ;;
        *)
            log_error "Système non supporté pour l'installation automatique de Git"
            exit 1
            ;;
    esac
    
    log_success "Git installé"
}

check_memory() {
    log_step "Vérification de la mémoire RAM..."
    
    if [[ "$OS" == "macos" ]]; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    log_info "Mémoire RAM détectée: ${MEMORY_GB}GB"
    
    if [ "$MEMORY_GB" -lt "$REQUIRED_MEMORY_GB" ]; then
        log_warning "Mémoire RAM faible (${MEMORY_GB}GB < ${REQUIRED_MEMORY_GB}GB recommandé)"
        log_info "Le système fonctionnera mais les performances seront limitées"
    else
        log_success "Mémoire RAM suffisante"
    fi
}

check_disk_space() {
    log_step "Vérification de l'espace disque..."
    
    AVAILABLE_GB=$(df -BG . | awk 'NR==2{print int($4)}')
    log_info "Espace disque disponible: ${AVAILABLE_GB}GB"
    
    if [ "$AVAILABLE_GB" -lt "$REQUIRED_DISK_GB" ]; then
        log_error "Espace disque insuffisant (${AVAILABLE_GB}GB < ${REQUIRED_DISK_GB}GB requis)"
        exit 1
    else
        log_success "Espace disque suffisant"
    fi
}

check_gpu() {
    log_step "Vérification GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        log_success "GPU NVIDIA détecté: $GPU_INFO"
        GPU_AVAILABLE=true
    elif command -v rocm-smi &> /dev/null; then
        log_success "GPU AMD ROCm détecté"
        GPU_AVAILABLE=true
    else
        log_info "Aucun GPU détecté - utilisation CPU uniquement"
        GPU_AVAILABLE=false
    fi
}

# Installation des dépendances système
install_system_dependencies() {
    log_step "Installation des dépendances système..."
    
    case $OS in
        ubuntu)
            sudo apt update
            sudo apt install -y \
                curl wget unzip \
                build-essential \
                libssl-dev libffi-dev \
                software-properties-common \
                htop tree vim
            ;;
        centos)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                curl wget unzip \
                openssl-devel libffi-devel \
                htop tree vim
            ;;
        fedora)
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y \
                curl wget unzip \
                openssl-devel libffi-devel \
                htop tree vim
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install curl wget unzip htop tree vim
            fi
            ;;
    esac
    
    log_success "Dépendances système installées"
}

# Clonage du repository
clone_repository() {
    log_step "Clonage du repository..."
    
    if [ -d "$PROJECT_NAME" ]; then
        log_info "Repository déjà présent, mise à jour..."
        cd "$PROJECT_NAME"
        git pull origin main
    else
        log_info "Clonage depuis GitHub..."
        git clone https://github.com/LeZelote01/BYJY-LLM.git "$PROJECT_NAME"
        cd "$PROJECT_NAME"
    fi
    
    log_success "Repository cloné/mis à jour"
}

# Installation des dépendances Python
install_python_dependencies() {
    log_step "Installation des dépendances Python..."
    
    # Créer un environnement virtuel
    if [ ! -d "venv" ]; then
        log_info "Création de l'environnement virtuel..."
        python3 -m venv venv
    fi
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Mettre à jour pip
    pip install --upgrade pip wheel setuptools
    
    # Installer PyTorch selon la disponibilité GPU
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Installation de PyTorch avec support GPU..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installation de PyTorch CPU uniquement..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Installer les autres dépendances
    log_info "Installation des autres dépendances..."
    pip install -r requirements.txt
    
    log_success "Dépendances Python installées"
}

# Configuration initiale
initial_configuration() {
    log_step "Configuration initiale..."
    
    # Créer les répertoires nécessaires
    mkdir -p logs models/{base,lora_weights,quantized} datasets/sources configs/{cloud,local}
    
    # Copier les configurations par défaut
    if [ ! -f "configs/local/user_config.json" ]; then
        cp configs/local/default_config.json configs/local/user_config.json
    fi
    
    # Créer le script de lancement
    create_launcher_script
    
    log_success "Configuration initiale terminée"
}

create_launcher_script() {
    log_info "Création du script de lancement..."
    
    cat > launch.sh << 'EOF'
#!/bin/bash
# Script de lancement pour LLaMA CyberSec

cd "$(dirname "$0")"

# Activer l'environnement virtuel
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Lancer l'interface
echo "🚀 Démarrage de LLaMA CyberSec..."
python local_deployment/local_interface.py

echo "👋 Arrêt de LLaMA CyberSec"
EOF
    
    chmod +x launch.sh
    
    # Version Windows
    cat > launch.bat << 'EOF'
@echo off
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo 🚀 Démarrage de LLaMA CyberSec...
python local_deployment\local_interface.py

echo 👋 Arrêt de LLaMA CyberSec
pause
EOF
    
    log_success "Scripts de lancement créés"
}

# Tests de validation
run_validation_tests() {
    log_step "Exécution des tests de validation..."
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Test d'importation des modules principaux
    python3 -c "
import torch, transformers, flask
print('✅ Modules principaux importés')
"
    
    # Test du système
    if [ -f "tests/validate_pipeline.py" ]; then
        python tests/validate_pipeline.py --quick
    fi
    
    log_success "Tests de validation terminés"
}

# Configuration post-installation
post_installation_setup() {
    log_step "Configuration post-installation..."
    
    # Créer le dataset initial
    source venv/bin/activate
    
    if [ -f "scripts/dataset_manager.py" ]; then
        log_info "Création du dataset initial..."
        python scripts/dataset_manager.py --create-initial
    fi
    
    # Proposer d'installer un modèle
    echo
    read -p "Voulez-vous installer un modèle maintenant ? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation d'un modèle de base..."
        python local_deployment/local_installer.py --model-source "https://huggingface.co/microsoft/DialoGPT-medium"
    fi
    
    log_success "Configuration post-installation terminée"
}

# Affichage du résumé final
show_completion_summary() {
    echo
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 INSTALLATION TERMINÉE AVEC SUCCÈS! 🎉${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}📁 Répertoire d'installation:${NC} $(pwd)"
    echo -e "${CYAN}🐍 Environnement virtuel:${NC} venv/"
    echo -e "${CYAN}📊 Dataset:${NC} datasets/"
    echo -e "${CYAN}🤖 Modèles:${NC} models/"
    echo
    
    echo -e "${YELLOW}🚀 DÉMARRAGE RAPIDE:${NC}"
    echo "1. Lancez l'interface:"
    echo "   ./launch.sh     (Linux/macOS)"
    echo "   launch.bat      (Windows)"
    echo
    echo "2. Ouvrez votre navigateur:"
    echo "   Interface web: http://localhost:8080"
    echo "   API REST: http://localhost:8081"
    echo
    
    echo -e "${YELLOW}📚 PROCHAINES ÉTAPES:${NC}"
    echo "1. 📊 Enrichir le dataset:"
    echo "   python scripts/dataset_manager.py --enrich-all"
    echo
    echo "2. ☁️  Entraîner sur le cloud:"
    echo "   python cloud_training/cloud_trainer.py"
    echo
    echo "3. 🏠 Installer plus de modèles:"
    echo "   python utils/model_manager.py --action list --show-remote"
    echo
    
    echo -e "${YELLOW}📖 DOCUMENTATION:${NC}"
    echo "• Guide complet: docs/GUIDE_UTILISATION.md"
    echo "• Configuration: docs/CONFIGURATION.md"
    echo "• API: docs/API.md"
    echo
    
    echo -e "${YELLOW}🆘 SUPPORT:${NC}"
    echo "• Issues GitHub: https://github.com/LeZelote01/BYJY-LLM/issues"
    echo "• Tests: python tests/validate_pipeline.py"
    echo "• Logs: logs/"
    echo
    
    echo -e "${GREEN}✨ Votre assistant IA cybersécurité est prêt à l'emploi! ✨${NC}"
}

# Configuration rapide interactive
interactive_setup() {
    echo
    echo -e "${CYAN}🔧 CONFIGURATION RAPIDE${NC}"
    echo "Voulez-vous configurer le système maintenant ?"
    echo
    echo "1. Configuration complète (recommandée)"
    echo "2. Configuration cloud uniquement"
    echo "3. Configuration locale uniquement"
    echo "4. Ignorer pour l'instant"
    echo
    
    read -p "Votre choix (1-4): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            log_info "Lancement de la configuration complète..."
            python quick_setup.py --auto complete
            ;;
        2)
            log_info "Lancement de la configuration cloud..."
            python quick_setup.py --auto cloud
            ;;
        3)
            log_info "Lancement de la configuration locale..."
            python quick_setup.py --auto local
            ;;
        4)
            log_info "Configuration ignorée - vous pourrez la lancer plus tard avec:"
            log_info "python quick_setup.py"
            ;;
        *)
            log_info "Choix invalide - configuration ignorée"
            ;;
    esac
}

# Fonction principale
main() {
    print_header
    
    log_info "Début de l'installation automatique..."
    
    # Vérifications système
    check_system_requirements
    
    # Installation des dépendances système
    install_system_dependencies
    
    # Clonage du repository
    clone_repository
    
    # Installation des dépendances Python
    install_python_dependencies
    
    # Configuration initiale
    initial_configuration
    
    # Tests de validation
    run_validation_tests
    
    # Configuration post-installation
    post_installation_setup
    
    # Configuration interactive (optionnelle)
    interactive_setup
    
    # Résumé final
    show_completion_summary
    
    log_success "Installation terminée avec succès!"
}

# Gestion des arguments
case "${1:-}" in
    --help|-h)
        echo "Installation automatique LLaMA-3-8B Cybersécurité"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Afficher cette aide"
        echo "  --no-gpu       Forcer installation CPU uniquement"
        echo "  --quick        Installation rapide sans interaction"
        echo "  --cloud-only   Configuration cloud uniquement"
        echo "  --local-only   Configuration locale uniquement"
        echo ""
        echo "Exemples:"
        echo "  $0                 # Installation interactive complète"
        echo "  $0 --quick         # Installation rapide automatique"
        echo "  $0 --no-gpu        # Installation sans support GPU"
        echo ""
        exit 0
        ;;
    --no-gpu)
        GPU_AVAILABLE=false
        main
        ;;
    --quick)
        INTERACTIVE=false
        main
        ;;
    --cloud-only)
        CLOUD_ONLY=true
        main
        ;;
    --local-only)
        LOCAL_ONLY=true
        main
        ;;
    *)
        main
        ;;
esac