# 🛡️ LLaMA-3-8B Cybersécurité - Architecture Cloud/Local

Système complet de fine-tuning LLaMA-3-8B spécialisé en cybersécurité avec **entraînement cloud** et **déploiement local**.

## 🚀 Architecture Innovante

### ☁️ **CLOUD** - Entraînement Haute Performance
- **Plateformes supportées** : Google Colab, AWS SageMaker, Azure ML, Paperspace
- **GPU puissants** : A100, V100, RTX series
- **Entraînement distribué** multi-GPU
- **Monitoring avancé** avec Weights & Biases
- **Stockage cloud** automatique (S3, GCS, Azure Blob)

### 🏠 **LOCAL** - Utilisation Optimisée  
- **CPU optimisé** : Fonctionne sur machines standard (8GB+ RAM)
- **GPU léger** : Support RTX/GTX pour accélération optionnelle
- **Interface web** moderne et intuitive
- **API REST** complète
- **Installation simple** en un clic

## 📁 Structure du Projet

```
/app/
├── 🌩️  cloud_training/           # Entraînement cloud
│   ├── cloud_trainer.py          # Trainer optimisé multi-plateforme
│   └── colab_notebook.ipynb      # Notebook Google Colab
├── 🏠 local_deployment/          # Déploiement local
│   ├── local_installer.py        # Installation automatique
│   └── local_interface.py        # Interface web + API
├── 🔧 utils/                     # Utilitaires
│   ├── cloud_uploader.py         # Upload/download cloud
│   └── model_manager.py          # Gestion des modèles
├── ⚙️  configs/                  # Configurations
│   ├── cloud/                    # Config cloud (Colab, AWS, etc.)
│   └── local/                    # Config locales
├── 📊 datasets/                  # Datasets cybersécurité
├── 🤖 models/                    # Modèles entraînés
├── 📚 docs/                      # Documentation
└── 🧪 tests/                     # Tests automatisés
```

## 🎯 Workflow Complet

### 1. 📝 Préparation du Dataset
```bash
# Créer le dataset initial (15+ exemples cybersécurité)
python scripts/dataset_manager.py --create-initial

# Enrichir automatiquement depuis sources publiques
python scripts/dataset_manager.py --enrich-all
```

### 2. ☁️ Entraînement Cloud

#### Google Colab (Recommandé - Gratuit)
```python
# Dans Google Colab
!git clone https://github.com/VotreRepo/BYJY-LLM.git
%cd BYJY-LLM

# Installation automatique
!python cloud_training/cloud_trainer.py \
  --model-name "meta-llama/Llama-2-7b-hf" \
  --dataset-url "https://votre-dataset.jsonl" \
  --wandb-project "llama-cybersec"
```

#### AWS SageMaker
```bash
python cloud_training/cloud_trainer.py \
  --model-name "meta-llama/Llama-2-7b-hf" \
  --dataset-path "./datasets/enriched_dataset.jsonl" \
  --cloud-bucket "s3://votre-bucket" \
  --epochs 3
```

### 3. 🏠 Installation Locale

#### Installation Automatique
```bash
# Windows
python local_deployment/local_installer.py \
  --model-source "https://huggingface.co/votre-modele-entraine"

# Linux/macOS  
python3 local_deployment/local_installer.py \
  --model-source "https://huggingface.co/votre-modele-entraine"
```

#### Lancement
```bash
# Windows
launch.bat

# Linux/macOS
./launch.sh
```

## ✨ Nouvelles Fonctionnalités

### 🌩️ Entraînement Cloud Avancé
- **Détection automatique** de plateforme (Colab, AWS, Azure)
- **Configuration adaptative** selon GPU disponible
- **Reprise automatique** d'entraînement en cas d'interruption
- **Upload automatique** vers stockage cloud
- **Monitoring temps réel** avec métriques avancées

### 🏠 Interface Locale Moderne
- **Interface web** responsive avec thème sombre
- **Chat intelligent** avec gestion de sessions
- **API REST** complète pour intégrations
- **Gestion de modèles** avec mise à jour automatique
- **Configuration visuelle** des paramètres

### 🔧 Outils de Gestion
- **Model Manager** : Installation/mise à jour automatique des modèles
- **Cloud Uploader** : Support S3, GCS, Azure Blob, Hugging Face
- **Monitoring avancé** : Logs, métriques, alertes
- **Tests automatisés** : Validation continue du système

## 🛠️ Installation Rapide

### Prérequis
- **Python 3.8+**
- **8GB RAM minimum** (16GB recommandé)
- **10GB espace disque**
- **Connexion Internet** pour téléchargement initial

### Option 1 : Installation Complète
```bash
# Cloner le repository
git clone https://github.com/LeZelote01/BYJY-LLM.git
cd BYJY-LLM

# Installation des dépendances
pip install -r requirements.txt

# Installation locale avec modèle
python local_deployment/local_installer.py
```

### Option 2 : Installation Express
```bash
# Script d'installation automatique
curl -fsSL https://raw.githubusercontent.com/LeZelote01/BYJY-LLM/main/install.sh | bash
```

## 🎮 Utilisation

### Interface Web
1. Lancez l'interface : `./launch.sh` ou `launch.bat`
2. Ouvrez votre navigateur : `http://localhost:8080`
3. Commencez à poser vos questions cybersécurité !

### API REST
```python
import requests

response = requests.post('http://localhost:8081/api/generate', json={
    "prompt": "Créer une règle YARA pour détecter un ransomware",
    "temperature": 0.7,
    "max_tokens": 512
})

print(response.json()['response'])
```

### Exemples d'Utilisation

#### 🔍 Création de Règles de Détection
```
Prompt: "Créer une règle Sigma pour détecter les tentatives de brute force RDP"
```

#### 📊 Analyse de Logs
```  
Prompt: "Analyser ce log Apache pour identifier une attaque : 
192.168.1.100 - - [10/Jan/2024:13:37:42] 'GET /admin.php?id=1' OR '1'='1'"
```

#### 🔒 Audit de Code
```
Prompt: "Identifier les vulnérabilités dans ce code Python et proposer des corrections"
```

## 🎛️ Configuration Avancée

### Paramètres de Modèle
```json
{
  "inference": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "context_length": 2048
  }
}
```

### Configuration Cloud
```json
{
  "cloud_training": {
    "platform": "auto",
    "use_4bit_quantization": true,
    "gradient_accumulation_steps": 16,
    "wandb_project": "llama-cybersec"
  }
}
```

## 📈 Performance

### Entraînement Cloud
- **Google Colab (T4)** : ~2-3 heures pour 3 époques
- **AWS SageMaker (p3.2xlarge)** : ~1-2 heures pour 3 époques  
- **GPU local (RTX 4090)** : ~45 minutes pour 3 époques

### Inférence Locale
- **CPU (16 cores)** : ~2-5 secondes par réponse
- **GPU (RTX 3080)** : ~0.5-1 seconde par réponse
- **GPU (RTX 4090)** : ~0.2-0.5 seconde par réponse

## 🔄 Gestion des Modèles

### Télécharger un Nouveau Modèle
```bash
python utils/model_manager.py --action install \
  --model-id "llama-cybersec-v2" \
  --source-url "https://huggingface.co/modele-v2"
```

### Mettre à Jour
```bash
python utils/model_manager.py --action update --model-id "llama-cybersec-v2"
```

### Lister les Modèles
```bash
python utils/model_manager.py --action list --show-remote
```

## 🧪 Tests et Validation

### Tests Automatisés
```bash
# Tests complets du système
python tests/validate_pipeline.py

# Tests spécifiques
pytest tests/test_system.py -v
```

### Validation de Performance
```bash
# Benchmark d'inférence
python tests/benchmark_inference.py

# Test de charge
python tests/load_test.py --concurrent-users 10
```

## 🔧 Dépannage

### Problèmes Fréquents

#### Modèle ne se charge pas
```bash
# Vérifier l'intégrité
python utils/model_manager.py --action info --model-id "votre-modele"

# Réinstaller si nécessaire
python utils/model_manager.py --action install --model-id "votre-modele" --force
```

#### Interface web inaccessible
```bash
# Vérifier le port
netstat -an | grep 8080

# Changer le port
python local_deployment/local_interface.py --port 8081
```

#### Erreur de mémoire
```json
// Ajuster dans config/local/default_config.json
{
  "inference": {
    "context_length": 1024,
    "batch_size": 1
  }
}
```

## 📚 Documentation Détaillée

- **Guide d'Installation** : [`docs/INSTALLATION.md`](docs/INSTALLATION.md)
- **Guide d'Utilisation** : [`docs/GUIDE_UTILISATION.md`](docs/GUIDE_UTILISATION.md)
- **API Reference** : [`docs/API.md`](docs/API.md)
- **Configuration** : [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- **Développeurs** : [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)

## 🤝 Contribution

Nous accueillons vos contributions ! Consultez [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) pour commencer.

### Roadmap
- [ ] Support des modèles multimodaux
- [ ] Interface vocale
- [ ] Plugin VS Code/JetBrains
- [ ] Support mobile (Android/iOS)
- [ ] Intégration Slack/Teams

## 📄 Licence

Ce projet est sous licence MIT. Voir [`LICENSE`](LICENSE) pour plus de détails.

## 🙋‍♂️ Support

- **Issues GitHub** : [Signaler un bug](https://github.com/LeZelote01/BYJY-LLM/issues)
- **Discussions** : [Forum communautaire](https://github.com/LeZelote01/BYJY-LLM/discussions)
- **Email** : support@llama-cybersec.com

## 📊 Statistiques

![GitHub stars](https://img.shields.io/github/stars/LeZelote01/BYJY-LLM)
![GitHub forks](https://img.shields.io/github/forks/LeZelote01/BYJY-LLM)
![GitHub issues](https://img.shields.io/github/issues/LeZelote01/BYJY-LLM)
![License](https://img.shields.io/github/license/LeZelote01/BYJY-LLM)

---

**🛡️ LLaMA CyberSec** - L'IA cybersécurité de nouvelle génération  
*Entraînement cloud, utilisation locale, performance maximale*