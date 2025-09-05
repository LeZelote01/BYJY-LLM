# Guide d'Utilisation - Système de Fine-tuning LLaMA-3-8B Cybersécurité

Ce guide vous accompagne dans l'utilisation complète du système de fine-tuning LLaMA-3-8B spécialisé en cybersécurité.

## Table des Matières

1. [Installation](#installation)
2. [Configuration Initiale](#configuration-initiale)
3. [Gestion des Datasets](#gestion-des-datasets)
4. [Entraînement du Modèle](#entraînement-du-modèle)
5. [Fusion et Quantisation](#fusion-et-quantisation)
6. [Déploiement Local](#déploiement-local)
7. [Utilisation du Modèle](#utilisation-du-modèle)
8. [Résolution de Problèmes](#résolution-de-problèmes)

## Installation

### Prérequis Système

- **OS** : Ubuntu 22.04 LTS (recommandé)
- **RAM** : 20 GB minimum (32 GB recommandé)
- **Stockage** : 100 GB d'espace libre
- **GPU** : NVIDIA avec 8GB+ VRAM (optionnel mais recommandé)
- **Python** : 3.10+

### Installation Automatique

```bash
# Rendre le script exécutable
chmod +x scripts/install.sh

# Lancer l'installation complète
./scripts/install.sh

# Installation sans CUDA (CPU uniquement)
./scripts/install.sh --no-cuda

# Installation Python uniquement
./scripts/install.sh --python-only
```

### Installation Manuelle

Si l'installation automatique échoue :

```bash
# 1. Installer Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# 2. Créer l'environnement virtuel
python3.10 -m venv llama_cybersec
source llama_cybersec/bin/activate

# 3. Installer PyTorch
# Pour GPU CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour CPU uniquement
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Installer les autres dépendances
pip install transformers datasets peft bitsandbytes accelerate
pip install requests beautifulsoup4 gitpython wandb
```

### Vérification de l'Installation

```bash
# Activer l'environnement
source activate_env.sh

# Vérifier les composants
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import peft; print('PEFT:', peft.__version__)"

# Test CUDA (si GPU)
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

## Configuration Initiale

### Démarrage Rapide

```bash
# Lancer la configuration initiale
./quick_start.sh
```

Ce script vous guidera à travers :
- Création du dataset initial
- Téléchargement des sources publiques (optionnel)
- Vérification de la configuration

### Configuration Manuelle

#### 1. Accès au Modèle LLaMA

Pour utiliser LLaMA-2 ou LLaMA-3 :

1. Créer un compte Hugging Face : https://huggingface.co/
2. Demander l'accès au modèle :
   - LLaMA-2-7B : https://huggingface.co/meta-llama/Llama-2-7b-hf
   - LLaMA-2-13B : https://huggingface.co/meta-llama/Llama-2-13b-hf
3. Installer et configurer Hugging Face CLI :
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

#### 2. Configuration des Répertoires

```bash
# Vérifier la structure
tree /app -L 2

# Créer les répertoires manquants si nécessaire
mkdir -p /app/{datasets,models,configs,logs,docs}
mkdir -p /app/models/{base,lora_weights,quantized}
mkdir -p /app/datasets/sources
```

## Gestion des Datasets

### Création du Dataset Initial

```bash
# Créer le dataset de base (15+ exemples)
python scripts/dataset_manager.py --create-initial

# Vérifier le dataset
python scripts/dataset_manager.py --stats
python scripts/dataset_manager.py --list 5
```

### Ajout d'Exemples Manuels

```bash
# Ajouter un exemple interactivement
python scripts/dataset_manager.py --add-example \
  "Créer une règle Sigma pour détecter les connexions RDP suspectes" \
  "Voici une règle Sigma pour détecter les connexions RDP suspectes..."
```

### Enrichissement Automatique

```bash
# Enrichir depuis toutes les sources publiques
python scripts/dataset_manager.py --enrich-all

# Cela télécharge et parse :
# - Règles Sigma (SigmaHQ/sigma)
# - Règles YARA (Yara-Rules/rules) 
# - Atomic Red Team
# - MITRE ATT&CK
# - Scripts de sécurité
# - Guides OWASP
```

### Ajout de Sources Locales

```bash
# Placer vos fichiers dans le répertoire sources
cp mes_regles_sigma.yml /app/datasets/sources/
cp mes_scripts_python.py /app/datasets/sources/
cp mes_logs.log /app/datasets/sources/

# Re-parser pour inclure les nouvelles sources
python scripts/dataset_manager.py --enrich-all
```

### Validation du Dataset

```bash
# Valider la structure JSON
python scripts/dataset_manager.py --validate

# Statistiques détaillées
python scripts/dataset_manager.py --stats
```

Exemple de sortie :
```
=== Statistiques du Dataset ===
Nombre total d'exemples: 847
Longueur moyenne instruction: 87 caractères
Longueur moyenne output: 456 caractères

Sujets couverts:
  sigma: 234 exemples
  yara: 156 exemples
  python: 123 exemples
  log: 98 exemples
  bash: 76 exemples
  ...
```

## Entraînement du Modèle

### Estimation du Temps d'Entraînement

```bash
# Estimer la durée avant de commencer
python scripts/train_lora.py --estimate-time
```

### Entraînement Basique

```bash
# Entraînement avec les paramètres par défaut
python scripts/train_lora.py

# Avec suivi Weights & Biases (optionnel)
wandb login
python scripts/train_lora.py
```

### Entraînement Personnalisé

```bash
# Ajuster les paramètres principaux
python scripts/train_lora.py \
  --model-name "meta-llama/Llama-2-7b-hf" \
  --dataset-path "/app/datasets/enriched_dataset.jsonl" \
  --output-dir "/app/models/lora_weights" \
  --epochs 3 \
  --batch-size 1 \
  --learning-rate 2e-4

# Pour système avec peu de RAM/GPU
python scripts/train_lora.py \
  --batch-size 1 \
  --no-4bit  # Désactive la quantisation si problème
```

### Surveillance de l'Entraînement

```bash
# Dans un autre terminal
tail -f /app/logs/training_*.log

# Monitoring GPU (si disponible)
watch -n 1 nvidia-smi

# Monitoring système
htop
```

### Arrêt et Reprise

L'entraînement sauvegarde des checkpoints automatiquement :

```bash
# En cas d'interruption, relancer le même commande
# Le training reprendra au dernier checkpoint
python scripts/train_lora.py
```

## Fusion et Quantisation

Une fois l'entraînement terminé :

### Fusion des Poids LoRA

```bash
# Fusion basique (q4_0 et q5_1)
python scripts/merge_and_quantize.py

# Fusion avec quantisations spécifiques
python scripts/merge_and_quantize.py \
  --quantization q4_0 q5_1 q8_0

# Fusion uniquement, sans quantisation
python scripts/merge_and_quantize.py --merge-only
```

### Types de Quantisation Disponibles

| Type | Taille | Qualité | Usage |
|------|---------|---------|-------|
| q2_k | ~2.6GB | Basse | Tests rapides |
| q4_0 | ~4.3GB | Bonne | **Recommandé général** |
| q5_1 | ~5.4GB | Très bonne | **Recommandé qualité** |
| q8_0 | ~8.5GB | Excellente | Si assez de RAM |
| f16  | ~13GB  | Maximale | GPU puissant |

### Installation de llama.cpp

Si pas encore installé :

```bash
# Sera installé automatiquement, ou manuellement :
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j4
```

## Déploiement Local

### Option 1 : Ollama (Recommandé)

```bash
# 1. Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Créer le modèle depuis le Modelfile généré
ollama create llama-cybersec -f /app/models/quantized/Modelfile

# 3. Tester
ollama run llama-cybersec "Créer une règle YARA pour détecter un ransomware"
```

### Option 2 : llama.cpp Direct

```bash
# Utiliser le script généré
/app/models/quantized/run_llama_cybersec.sh

# Ou directement
~/llama.cpp/main \
  -m /app/models/quantized/llama-cybersec-q4_0.gguf \
  -p "### Instruction:\nAnalyser ce log Apache\n\n### Response:\n" \
  -n 256 --temp 0.7
```

### Option 3 : API REST avec Ollama

```bash
# Démarrer Ollama en mode serveur
ollama serve

# Utiliser l'API
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-cybersec",
    "prompt": "Expliquer les techniques de lateral movement",
    "stream": false
  }'
```

## Utilisation du Modèle

### Exemples de Prompts

#### Création de Règles Sigma
```
Instruction: Créer une règle Sigma pour détecter les tentatives de brute force RDP

Response: Voici une règle Sigma pour détecter les tentatives de brute force RDP :

title: RDP Brute Force Detection
status: experimental
logsource:
    product: windows
    service: security
detection:
    selection:
        EventID: 4625
        LogonType: 10
        Source: '*'
    timeframe: 5m
    condition: selection | count() > 5

Cette règle détecte plus de 5 échecs de connexion RDP en 5 minutes...
```

#### Analyse de Logs
```
Instruction: Analyser ce log Apache : 192.168.1.100 - - [10/Jan/2024:13:37:42 +0000] "GET /admin.php?id=1' OR '1'='1 HTTP/1.1" 200 1234

Response: Analyse du log Apache :

**Type d'attaque détecté** : Injection SQL
**Indicateurs** :
- Pattern d'injection : OR '1'='1
- Cible : /admin.php avec paramètre id
- Code de réponse : 200 (succès - préoccupant)

**Recommandations** :
- Bloquer l'IP 192.168.1.100
- Vérifier les paramètres de admin.php
- Implémenter une validation d'entrée...
```

### Formats de Prompt Supportés

Le modèle est entraîné avec ce format :
```
### Instruction:
[Votre question ou demande]

### Response:
[La réponse sera générée ici]
```

### Cas d'Usage Principaux

1. **Création de règles de détection**
   - Règles Sigma, YARA, Snort
   - Configurations IDS/IPS

2. **Analyse de logs de sécurité**
   - Logs Apache/Nginx
   - Windows Event Logs
   - Logs de sécurité système

3. **Développement de scripts**
   - Scripts Python d'analyse
   - Scripts Bash de monitoring
   - Outils de pentesting

4. **Analyse de malware**
   - Identification de techniques
   - IOCs (Indicators of Compromise)
   - Recommandations de mitigation

5. **Formation et documentation**
   - Explications techniques
   - Guides de bonnes pratiques
   - Scénarios d'incident response

## Résolution de Problèmes

### Problèmes d'Installation

#### Erreur CUDA
```bash
# Vérifier la version CUDA
nvidia-smi
nvcc --version

# Réinstaller PyTorch pour la bonne version CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Problème de Mémoire
```bash
# Réduire la taille de batch
python scripts/train_lora.py --batch-size 1

# Utiliser la quantisation 4-bit
python scripts/train_lora.py  # (activée par défaut)

# Monitoring mémoire
watch -n 1 'free -h && nvidia-smi'
```

### Problèmes d'Entraînement

#### Entraînement Lent
- Vérifier que CUDA est utilisé
- Augmenter le batch size si assez de mémoire
- Utiliser plusieurs GPUs si disponibles

#### Perte NaN ou Infinie
- Réduire le learning rate (--learning-rate 1e-4)
- Vérifier la qualité du dataset
- Utiliser FP16 (activé par défaut)

#### Arrêt Inattendu
- Vérifier l'espace disque disponible
- Surveiller la température GPU
- Utiliser tmux pour les longs entraînements

### Problèmes de Quantisation

#### llama.cpp Non Trouvé
```bash
# Installation manuelle
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j$(nproc)
```

#### Erreur de Conversion
- Vérifier que les poids LoRA sont complets
- S'assurer d'avoir assez d'espace disque
- Utiliser la version f16 en premier

### Problèmes de Déploiement

#### Ollama ne Démarre Pas
```bash
# Vérifier l'installation
ollama --version

# Réinstaller si nécessaire
curl -fsSL https://ollama.ai/install.sh | sh

# Vérifier les permissions
sudo systemctl status ollama
```

#### Modèle Lent en Inférence
- Utiliser une quantisation plus légère (q4_0)
- Augmenter le nombre de threads CPU
- Vérifier la disponibilité GPU

### Logs et Diagnostic

```bash
# Logs d'entraînement
tail -f /app/logs/training_*.log

# Logs de quantisation
tail -f /app/logs/merge_quantize_*.log

# Logs système
journalctl -u ollama -f

# Monitoring des ressources
htop
nvidia-smi
df -h
```

### Support et Communauté

En cas de problème persistant :

1. Vérifier les logs détaillés
2. Consulter la documentation Hugging Face Transformers
3. Vérifier les issues GitHub des projets utilisés
4. Tester avec un dataset plus petit
5. Vérifier la compatibilité des versions

### Configuration Recommandée par Type de Machine

#### Machine Haute Performance (32GB RAM, RTX 4090)
```bash
python scripts/train_lora.py \
  --model-name "meta-llama/Llama-2-13b-hf" \
  --batch-size 2 \
  --epochs 5 \
  --learning-rate 2e-4
```

#### Machine Standard (20GB RAM, RTX 3080)
```bash
python scripts/train_lora.py \
  --model-name "meta-llama/Llama-2-7b-hf" \
  --batch-size 1 \
  --epochs 3 \
  --learning-rate 2e-4
```

#### Machine Légère (16GB RAM, CPU)
```bash
python scripts/train_lora.py \
  --model-name "meta-llama/Llama-2-7b-hf" \
  --batch-size 1 \
  --epochs 2 \
  --no-4bit \
  --learning-rate 1e-4
```

---

Ce guide couvre l'utilisation complète du système. Pour des questions spécifiques, consultez les fichiers de configuration dans `/app/configs/` et les logs dans `/app/logs/`.