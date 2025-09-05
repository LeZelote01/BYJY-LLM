# 📚 Guide d'Utilisation - LLaMA CyberSec v2.0

Guide complet pour utiliser le système LLaMA-3-8B Cybersécurité avec architecture cloud/local.

## 📋 Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Installation Rapide](#installation-rapide)
3. [Entraînement Cloud](#entraînement-cloud)
4. [Déploiement Local](#déploiement-local)
5. [Interface Utilisateur](#interface-utilisateur)
6. [API REST](#api-rest)
7. [Gestion des Modèles](#gestion-des-modèles)
8. [Cas d'Usage](#cas-dusage)
9. [Configuration Avancée](#configuration-avancée)
10. [Dépannage](#dépannage)

## 🎯 Vue d'Ensemble

### Architecture Hybride

Le système LLaMA CyberSec utilise une architecture innovante **Cloud/Local** :

- **☁️ CLOUD** : Entraînement sur GPU puissants (Google Colab, AWS, Azure)
- **🏠 LOCAL** : Utilisation optimisée sur votre machine (CPU/GPU léger)

### Avantages Clés

✅ **Économique** : Entraînement gratuit sur Google Colab  
✅ **Performant** : GPU cloud A100/V100 pour l'entraînement  
✅ **Privé** : Données sensibles restent locales  
✅ **Portable** : Fonctionne sur laptop standard (8GB RAM)  
✅ **Évolutif** : Mise à jour des modèles automatique  

## 🚀 Installation Rapide

### Prérequis Système

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **OS** | Windows 10, Ubuntu 18.04, macOS 10.15 | Windows 11, Ubuntu 22.04, macOS 12+ |
| **RAM** | 8 GB | 16 GB |
| **Stockage** | 10 GB libre | 50 GB libre |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Optionnel | GTX 1660+ ou RTX série |
| **Python** | 3.8+ | 3.10+ |

### Installation Automatique

#### Option 1 : Script d'Installation
```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/LeZelote01/BYJY-LLM/main/install.sh | bash

# Windows (PowerShell en admin)
iwr -useb https://raw.githubusercontent.com/LeZelote01/BYJY-LLM/main/install.ps1 | iex
```

#### Option 2 : Installation Manuelle
```bash
# 1. Cloner le repository
git clone https://github.com/LeZelote01/BYJY-LLM.git
cd BYJY-LLM

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Vérifier l'installation
python tests/validate_pipeline.py --quick

# 4. Lancer l'installateur local
python local_deployment/local_installer.py
```

### Vérification de l'Installation

```bash
# Test des composants
python -c "import torch, transformers, flask; print('✅ Dépendances OK')"

# Test du système
python tests/test_system.py

# Lancement de l'interface
python local_deployment/local_interface.py
```

## ☁️ Entraînement Cloud

### 1. Google Colab (Gratuit - Recommandé)

#### Préparation
1. Ouvrez [Google Colab](https://colab.research.google.com/)
2. Activez le GPU : `Runtime` → `Change runtime type` → `GPU`
3. Vérifiez le GPU : `!nvidia-smi`

#### Entraînement
```python
# Cellule 1 : Installation
!git clone https://github.com/LeZelote01/BYJY-LLM.git
%cd BYJY-LLM
!pip install -r requirements.txt

# Cellule 2 : Configuration
import os
os.environ['WANDB_API_KEY'] = 'votre_cle_wandb'  # Optionnel

# Cellule 3 : Entraînement
!python cloud_training/cloud_trainer.py \
  --model-name "meta-llama/Llama-2-7b-hf" \
  --dataset-path "./datasets/initial_dataset.jsonl" \
  --wandb-project "llama-cybersec-colab" \
  --epochs 3

# Cellule 4 : Téléchargement du modèle
from google.colab import files
!zip -r model_trained.zip /content/drive/MyDrive/LLaMA_CyberSec/models/
files.download('model_trained.zip')
```

#### Optimisations Colab
```python
# Configuration GPU T4 (15GB VRAM)
training_config = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "fp16": True,
    "dataloader_num_workers": 2
}

# Configuration GPU V100 (16GB VRAM) 
training_config = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "fp16": True,
    "dataloader_num_workers": 4
}
```

### 2. AWS SageMaker

#### Préparation
```bash
# Configuration AWS CLI
aws configure
pip install sagemaker boto3

# Variables d'environnement
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ROLE=arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
```

#### Entraînement
```python
# Script Python pour SageMaker
import sagemaker
from sagemaker.pytorch import PyTorch

# Configuration
role = sagemaker.get_execution_role()
sess = sagemaker.Session()

# Estimateur PyTorch
estimator = PyTorch(
    entry_point='cloud_trainer.py',
    source_dir='cloud_training',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'model-name': 'meta-llama/Llama-2-7b-hf',
        'epochs': 3,
        'learning-rate': 2e-4
    },
    use_spot_instances=True,
    max_wait=7200,
    checkpoint_s3_uri='s3://votre-bucket/checkpoints'
)

# Lancer l'entraînement
estimator.fit({'training': 's3://votre-bucket/dataset.jsonl'})
```

### 3. Azure Machine Learning

#### Préparation
```bash
# Installation Azure ML SDK
pip install azureml-sdk

# Configuration
az login
az account set --subscription "Votre-Subscription-ID"
```

#### Entraînement
```python
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

# Workspace
ws = Workspace.from_config()

# Configuration de calcul
compute_target = ComputeTarget(workspace=ws, name='gpu-cluster')

# Configuration du script
script_config = ScriptRunConfig(
    source_directory='cloud_training',
    script='cloud_trainer.py',
    arguments=[
        '--model-name', 'meta-llama/Llama-2-7b-hf',
        '--epochs', '3'
    ],
    compute_target=compute_target
)

# Lancer l'expérience
experiment = Experiment(ws, 'llama-cybersec')
run = experiment.submit(script_config)
```

## 🏠 Déploiement Local

### Installation du Modèle Entraîné

#### Depuis URL
```bash
python local_deployment/local_installer.py \
  --model-source "https://huggingface.co/votre-modele/resolve/main/model.gguf"
```

#### Depuis Fichier Local
```bash
python local_deployment/local_installer.py \
  --model-source "/chemin/vers/votre/model.gguf"
```

#### Depuis Cloud Storage
```bash
# AWS S3
python local_deployment/local_installer.py \
  --model-source "s3://votre-bucket/model.gguf"

# Google Cloud Storage  
python local_deployment/local_installer.py \
  --model-source "gs://votre-bucket/model.gguf"
```

### Configuration Système

Le système crée automatiquement la structure suivante :

#### Windows
```
C:\Users\VotreNom\AppData\Local\LLaMA-CyberSec\
├── models/
│   └── gguf/
├── config/
├── logs/
├── web/
└── conversations/
```

#### Linux/macOS
```
~/.llama-cybersec/
├── models/
│   └── gguf/
├── config/
├── logs/
├── web/
└── conversations/
```

### Lancement

#### Lancement Simple
```bash
# Windows
launch.bat

# Linux/macOS
./launch.sh
```

#### Lancement avec Options
```bash
python local_deployment/local_interface.py \
  --host 0.0.0.0 \
  --port 8080 \
  --config ./config/custom_config.json
```

#### Service Système (Linux)
```bash
# Créer le service
sudo tee /etc/systemd/system/llama-cybersec.service > /dev/null <<EOF
[Unit]
Description=LLaMA CyberSec Local Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/.llama-cybersec
ExecStart=/usr/bin/python3 local_deployment/local_interface.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Activer et démarrer
sudo systemctl enable llama-cybersec
sudo systemctl start llama-cybersec
```

## 🖥️ Interface Utilisateur

### Interface Web

Accédez à `http://localhost:8080` pour l'interface web moderne.

#### Fonctionnalités Principales

1. **Chat Intelligent**
   - Gestion des sessions
   - Historique des conversations
   - Streaming en temps réel
   - Format markdown

2. **Configuration Visuelle**
   - Température : 0.1 - 1.0
   - Longueur max : 100 - 2000 tokens
   - Top-p : 0.1 - 1.0
   - Thème sombre/clair

3. **Exemples Prédéfinis**
   - Règles YARA/Sigma/Snort
   - Analyse de logs
   - Audit de code
   - Techniques d'attaque

4. **Gestion des Sessions**
   - Sauvegarde automatique
   - Export/Import conversations
   - Recherche dans l'historique

#### Raccourcis Clavier
- `Ctrl + Enter` : Envoyer message
- `Ctrl + L` : Effacer la conversation
- `Ctrl + S` : Sauvegarder conversation
- `Esc` : Interrompre génération

### Interface en ligne de commande

```bash
# Mode interactif
python -c "
from local_deployment.local_interface import LocalLLaMAInterface
interface = LocalLLaMAInterface()
while True:
    prompt = input('🛡️ Votre question: ')
    if prompt.lower() in ['quit', 'exit']: break
    response = interface.generate_response(prompt)
    print(f'🤖 Réponse: {response}')
"

# Mode script
python -c "
from local_deployment.local_interface import LocalLLaMAInterface
interface = LocalLLaMAInterface()
response = interface.generate_response('Créer une règle YARA pour détecter un trojan')
print(response)
"
```

## 🔌 API REST

### Endpoints Disponibles

#### 1. Génération de Texte
```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "Créer une règle Sigma pour détecter les tentatives de brute force SSH",
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 0.9,
  "session_id": "optional-session-id"
}
```

**Réponse :**
```json
{
  "response": "title: SSH Brute Force Detection\nstatus: experimental\n...",
  "session_id": "generated-session-id",
  "timestamp": "2024-01-15T10:30:00Z",
  "tokens_generated": 245,
  "generation_time": 1.2
}
```

#### 2. Status du Système
```http
GET /api/status
```

**Réponse :**
```json
{
  "status": "online",
  "model_loaded": true,
  "model_name": "llama-cybersec-q4_0",
  "sessions": 3,
  "uptime": 3600,
  "memory_usage": {
    "total": "16GB",
    "used": "4.2GB",
    "available": "11.8GB"
  },
  "gpu_info": {
    "available": true,
    "name": "NVIDIA RTX 3080",
    "memory_used": "2.1GB",
    "memory_total": "10GB"
  }
}
```

#### 3. Gestion des Sessions
```http
# Récupérer une session
GET /api/session/{session_id}

# Supprimer une session
DELETE /api/session/{session_id}

# Lister toutes les sessions
GET /api/sessions
```

#### 4. Gestion des Modèles
```http
# Lister les modèles
GET /api/models

# Changer de modèle
POST /api/models/switch
{
  "model_id": "llama-cybersec-v2"
}

# Informations sur un modèle
GET /api/models/{model_id}
```

### Exemples d'Utilisation

#### Python
```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:8081"
session_id = None

def ask_cybersec_question(question):
    global session_id
    
    response = requests.post(f"{BASE_URL}/api/generate", json={
        "prompt": question,
        "temperature": 0.7,
        "max_tokens": 512,
        "session_id": session_id
    })
    
    if response.status_code == 200:
        data = response.json()
        session_id = data.get("session_id")
        return data["response"]
    else:
        return f"Erreur: {response.status_code}"

# Utilisation
print(ask_cybersec_question("Expliquer les techniques de lateral movement"))
print(ask_cybersec_question("Donner des exemples concrets"))
```

#### JavaScript/Node.js
```javascript
const axios = require('axios');

class CyberSecAI {
    constructor(baseUrl = 'http://localhost:8081') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }
    
    async ask(question, options = {}) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/generate`, {
                prompt: question,
                temperature: options.temperature || 0.7,
                max_tokens: options.maxTokens || 512,
                session_id: this.sessionId
            });
            
            this.sessionId = response.data.session_id;
            return response.data.response;
        } catch (error) {
            throw new Error(`API Error: ${error.response?.data?.error || error.message}`);
        }
    }
    
    async getStatus() {
        const response = await axios.get(`${this.baseUrl}/api/status`);
        return response.data;
    }
}

// Utilisation
const ai = new CyberSecAI();

(async () => {
    const answer = await ai.ask("Créer une règle YARA pour détecter un keylogger");
    console.log(answer);
})();
```

#### curl
```bash
# Question simple
curl -X POST http://localhost:8081/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyser ce hash MD5: 5d41402abc4b2a76b9719d911017c592",
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Avec session
curl -X POST http://localhost:8081/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Continuer l'\''analyse",
    "session_id": "session-id-from-previous-response"
  }'
```

## 🤖 Gestion des Modèles

### Model Manager

Le système inclut un gestionnaire de modèles avancé pour télécharger, installer, et mettre à jour les modèles.

#### Lister les Modèles Disponibles
```bash
# Modèles locaux
python utils/model_manager.py --action list

# Inclure les modèles distants
python utils/model_manager.py --action list --show-remote
```

#### Installer un Modèle
```bash
# Depuis Hugging Face
python utils/model_manager.py --action install \
  --model-id "hf_llama-cybersec-v2" 

# Depuis URL directe
python utils/model_manager.py --action install \
  --model-id "custom-model" \
  --source-url "https://example.com/model.gguf"

# Forcer la réinstallation
python utils/model_manager.py --action install \
  --model-id "llama-cybersec" --force
```

#### Mettre à Jour un Modèle
```bash
python utils/model_manager.py --action update \
  --model-id "llama-cybersec"
```

#### Informations sur un Modèle
```bash
python utils/model_manager.py --action info \
  --model-id "llama-cybersec"
```

#### Désinstaller un Modèle
```bash
python utils/model_manager.py --action uninstall \
  --model-id "old-model"
```

#### Nettoyage
```bash
# Nettoyer les fichiers temporaires
python utils/model_manager.py --action cleanup
```

### Sources de Modèles

#### Modèles Officiels
- **llama-cybersec-base** : Modèle de base optimisé
- **llama-cybersec-advanced** : Version avancée avec plus de données
- **llama-cybersec-quantized** : Version quantifiée (plus rapide)

#### Modèles Communautaires
- **llama-cybersec-malware** : Spécialisé analyse malware
- **llama-cybersec-network** : Focus sécurité réseau
- **llama-cybersec-forensics** : Orienté investigations numériques

#### Configuration Automatique

Le système choisit automatiquement le meilleur modèle selon :
- **Mémoire disponible** : Modèle quantifié si RAM < 16GB
- **GPU disponible** : Version optimisée GPU si détecté
- **Cas d'usage** : Modèle spécialisé selon le domaine

## 💼 Cas d'Usage

### 1. 🔍 Création de Règles de Détection

#### Règles Sigma
```
Prompt: "Créer une règle Sigma pour détecter les tentatives d'élévation de privilèges via UAC bypass"

Réponse:
title: UAC Bypass Detection
status: experimental
description: Détection des tentatives de contournement de l'UAC Windows
references:
    - https://attack.mitre.org/techniques/T1548/002/
logsource:
    product: windows
    service: security
detection:
    selection:
        EventID: 4688
        NewProcessName|endswith:
            - '\fodhelper.exe'
            - '\computerdefaults.exe'
            - '\eventvwr.exe'
        ParentProcessName|endswith: '\cmd.exe'
    condition: selection
falsepositives:
    - Administration légitime
level: high
tags:
    - attack.privilege_escalation
    - attack.t1548.002
```

#### Règles YARA
```
Prompt: "Créer une règle YARA pour détecter les backdoors PowerShell"

Réponse:
rule PowerShell_Backdoor_Generic {
    meta:
        author = "LLaMA CyberSec"
        description = "Détection générique des backdoors PowerShell"
        date = "2024-01-15"
        version = "1.0"
        
    strings:
        $powershell = "powershell" nocase
        $encoded = "-EncodedCommand" nocase
        $bypass = "-ExecutionPolicy Bypass" nocase
        $hidden = "-WindowStyle Hidden" nocase
        $webclient = "System.Net.WebClient" nocase
        $downloadstring = "DownloadString" nocase
        $invoke = "IEX" nocase
        $base64_pattern = /[A-Za-z0-9+\/]{50,}={0,2}/
        
    condition:
        $powershell and (
            ($encoded and $base64_pattern) or
            ($bypass and $hidden and $webclient) or
            ($downloadstring and $invoke)
        )
}
```

#### Règles Snort
```
Prompt: "Créer une règle Snort pour détecter les attaques par SQL injection"

Réponse:
# Détection SQL Injection - Union Based
alert tcp any any -> $HOME_NET $HTTP_PORTS (
    msg:"SQL Injection Union Attack Detected";
    flow:to_server,established;
    content:"GET"; http_method;
    content:" union "; http_uri; nocase;
    content:" select "; http_uri; nocase;
    pcre:"/union\s+.*select/Ui";
    classtype:web-application-attack;
    sid:1000001;
    rev:1;
)

# Détection SQL Injection - Error Based  
alert tcp any any -> $HOME_NET $HTTP_PORTS (
    msg:"SQL Injection Error Based Attack";
    flow:to_server,established;
    content:"GET"; http_method;
    pcre:"/(\x27|\x22).*(or|and)\s*\d+\s*=\s*\d+/Ui";
    content:!"content-type"; http_header;
    classtype:web-application-attack;
    sid:1000002;
    rev:1;
)
```

### 2. 📊 Analyse de Logs

#### Logs Apache/Nginx
```
Prompt: "Analyser ce log Apache et identifier les menaces :
192.168.1.100 - - [15/Jan/2024:10:30:42 +0000] 'GET /admin/../../../etc/passwd HTTP/1.1' 200 1024"

Réponse:
🚨 ANALYSE DE LOG APACHE - MENACE DÉTECTÉE 🚨

**Type d'attaque identifié** : Path Traversal (Directory Traversal)

**Indicateurs malveillants** :
1. **Pattern d'attaque** : `../../../etc/passwd`
2. **Cible** : Fichier système sensible `/etc/passwd`
3. **Méthode** : Tentative d'accès via contournement de répertoire

**Détails techniques** :
- IP source : 192.168.1.100
- Timestamp : 15/Jan/2024:10:30:42 +0000
- Code de réponse : 200 (CRITIQUE - accès réussi!)
- Taille réponse : 1024 bytes (taille typique du fichier passwd)

**Actions recommandées** :
1. 🔥 **URGENT** : Bloquer immédiatement l'IP 192.168.1.100
2. 🔍 Vérifier l'intégrité du fichier /etc/passwd
3. 📝 Auditer les logs pour autres tentatives de cette IP
4. 🛡️ Mettre à jour les règles WAF pour bloquer les path traversal
5. 🔧 Vérifier la configuration Apache (disable directory indexing)

**Règle de détection suggérée** :
```
RewriteEngine On
RewriteCond %{THE_REQUEST} \.\./|\.\.\\
RewriteRule .* - [F,L]
```
```

#### Logs Windows Event
```
Prompt: "Analyser ces événements Windows pour détecter une intrusion :
EventID 4624, Logon Type 10, Account: admin, Source: 192.168.1.50, Time: 02:30:00
EventID 4648, Account: admin, Target: DOMAIN\svc_sql, Source: WORKSTATION01, Time: 02:31:00
EventID 4672, Account: admin, Privileges: SeDebugPrivilege, Time: 02:31:30"

Réponse:
🔍 ANALYSE ÉVÉNEMENTS WINDOWS - ACTIVITÉ SUSPECTE DÉTECTÉE

**Scénario d'attaque identifié** : Lateral Movement avec Élévation de Privilèges

**Timeline d'attaque** :
```
02:30:00 → Connexion RDP suspecte (EventID 4624)
02:31:00 → Usurpation d'identité (EventID 4648) 
02:31:30 → Privilèges debug acquis (EventID 4672)
```

**Analyse détaillée** :

1. **EventID 4624 - Connexion RDP (02:30:00)**
   - Logon Type 10 = RDP/Terminal Services
   - Heure anormale : 02:30 (hors horaires de travail)
   - Source externe : 192.168.1.50
   - ⚠️ **Alerte** : Connexion admin la nuit depuis IP non autorisée

2. **EventID 4648 - Usurpation d'identité (02:31:00)**
   - Compte source : admin
   - Compte cible : DOMAIN\svc_sql (service account)
   - Source : WORKSTATION01
   - 🚨 **Critique** : Tentative d'accès aux services SQL avec credentials différents

3. **EventID 4672 - Privilèges spéciaux (02:31:30)**
   - Privilege : SeDebugPrivilege (très sensible)
   - Permet l'accès mémoire d'autres processus
   - 🔥 **Alerte maximale** : Capacité de dump mémoire/passwords

**Techniques MITRE ATT&CK identifiées** :
- T1078 : Valid Accounts (utilisation compte admin)
- T1021.001 : Remote Desktop Protocol
- T1134 : Access Token Manipulation  
- T1055 : Process Injection (via SeDebugPrivilege)

**Actions de réponse immédiates** :
1. 🚨 Isoler WORKSTATION01 du réseau
2. 🔒 Désactiver le compte admin temporairement
3. 🔄 Changer le mot de passe svc_sql
4. 📊 Analyser les connexions SQL depuis 02:31:00
5. 🧪 Effectuer un scan antimalware complet
6. 📝 Examiner les logs SQL Server pour activités suspectes
```

### 3. 🔒 Audit de Code

#### Code Python Vulnérable
```
Prompt: "Auditer ce code Python et proposer des corrections :

import sqlite3
from flask import Flask, request

app = Flask(__name__)

@app.route('/user')
def get_user():
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f'SELECT * FROM users WHERE id = {user_id}'
    result = cursor.execute(query).fetchone()
    return str(result)"

Réponse:
🔍 AUDIT DE SÉCURITÉ - CODE PYTHON VULNÉRABLE

**❌ VULNÉRABILITÉS CRITIQUES IDENTIFIÉES :**

### 1. 🚨 SQL INJECTION (CWE-89)
**Problème** : Construction de requête par concaténation
```python
query = f'SELECT * FROM users WHERE id = {user_id}'  # DANGEREUX!
```

**Exploitation** :
```
GET /user?id=1 OR 1=1 --
→ Récupération de tous les utilisateurs

GET /user?id=1; DROP TABLE users; --  
→ Suppression de la table
```

**✅ CORRECTION** :
```python
# Utiliser des requêtes préparées
query = "SELECT * FROM users WHERE id = ?"
result = cursor.execute(query, (user_id,)).fetchone()
```

### 2. 🔓 INFORMATION DISCLOSURE (CWE-200)
**Problème** : Exposition directe des données sensibles
```python
return str(result)  # Expose potentiellement mots de passe, emails, etc.
```

**✅ CORRECTION** :
```python
# Filtrer les données sensibles
if result:
    safe_data = {
        'id': result[0],
        'username': result[1], 
        'email': result[2][:3] + "***"  # Masquer l'email
    }
    return jsonify(safe_data)
else:
    return jsonify({'error': 'User not found'}), 404
```

### 3. ⚠️ INPUT VALIDATION (CWE-20)  
**Problème** : Aucune validation des entrées utilisateur

**✅ CORRECTION** :
```python
# Validation du paramètre
user_id = request.args.get('id')
if not user_id or not user_id.isdigit():
    return jsonify({'error': 'Invalid user ID'}), 400

user_id = int(user_id)
if user_id <= 0:
    return jsonify({'error': 'User ID must be positive'}), 400
```

### 4. 🔒 ERROR HANDLING (CWE-209)
**Problème** : Gestion d'erreurs inexistante

**✅ CORRECTION** :
```python
try:
    # ... code de base de données ...
except sqlite3.Error as e:
    logger.error(f"Database error: {e}")
    return jsonify({'error': 'Database error'}), 500
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return jsonify({'error': 'Internal server error'}), 500
finally:
    if conn:
        conn.close()
```

**📋 CODE CORRIGÉ COMPLET** :
```python
import sqlite3
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/user')
def get_user():
    # 1. Validation des entrées
    user_id = request.args.get('id')
    if not user_id or not user_id.isdigit():
        return jsonify({'error': 'Invalid user ID format'}), 400
    
    user_id = int(user_id)
    if user_id <= 0:
        return jsonify({'error': 'User ID must be positive'}), 400
    
    conn = None
    try:
        # 2. Connexion sécurisée
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row  # Pour des résultats nommés
        cursor = conn.cursor()
        
        # 3. Requête préparée (protection SQL injection)
        query = "SELECT id, username, email FROM users WHERE id = ?"
        result = cursor.execute(query, (user_id,)).fetchone()
        
        # 4. Réponse sécurisée
        if result:
            user_data = {
                'id': result['id'],
                'username': result['username'],
                'email': result['email'][:3] + "***@" + result['email'].split('@')[1]
            }
            return jsonify(user_data)
        else:
            return jsonify({'error': 'User not found'}), 404
            
    except sqlite3.Error as e:
        logger.error(f"Database error for user_id {user_id}: {e}")
        return jsonify({'error': 'Database error'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    app.run(debug=False)  # Désactiver debug en production
```

**🛡️ MESURES SUPPLÉMENTAIRES RECOMMANDÉES :**
1. **Rate Limiting** : Limite les requêtes par IP
2. **Authentication** : Vérifier les permissions utilisateur  
3. **HTTPS** : Chiffrer les communications
4. **WAF** : Firewall applicatif
5. **Logging** : Enregistrer toutes les tentatives d'accès
6. **Monitoring** : Alertes sur activités suspectes

**📊 SCORE DE SÉCURITÉ :**
- Code original : 2/10 (Critique)
- Code corrigé : 8/10 (Bon niveau)
```

### 4. 🕵️ Investigation d'Incident

#### Analyse de Compromission
```
Prompt: "J'ai détecté des connexions suspectes depuis 192.168.1.100. Aidez-moi à mener l'investigation"

Réponse:
🔍 GUIDE D'INVESTIGATION - CONNEXIONS SUSPECTES

**📋 PHASE 1 : COLLECTE D'INFORMATIONS INITIALES**

### 1. Identification de l'IP Source
```bash
# Whois de l'IP
whois 192.168.1.100

# Géolocalisation  
curl -s "http://ip-api.com/json/192.168.1.100"

# Réputation de l'IP
curl -s "https://api.abuseipdb.com/api/v2/check" \
  -H "Key: YOUR_API_KEY" \
  -H "Accept: application/json" \
  --data-urlencode "ipAddress=192.168.1.100"
```

### 2. Analyse des Logs Réseau
```bash  
# Connexions actuelles
netstat -an | grep 192.168.1.100

# Historique des connexions (si disponible)
grep "192.168.1.100" /var/log/auth.log
grep "192.168.1.100" /var/log/secure
```

**📊 PHASE 2 : ANALYSE DÉTAILLÉE**

### 3. Timeline des Événements
```bash
# Logs système par date
journalctl --since "2024-01-15 00:00:00" | grep "192.168.1.100"

# Logs de connexion SSH
grep "192.168.1.100" /var/log/auth.log | tail -20

# Logs Apache/Nginx
grep "192.168.1.100" /var/log/apache2/access.log
grep "192.168.1.100" /var/log/nginx/access.log
```

### 4. Analyse des Services Exposés
```bash
# Ports ouverts
nmap -sS -O 192.168.1.100

# Services actifs sur la machine locale
ss -tuln | grep LISTEN

# Processus suspects
ps aux | grep -v "grep\|ps aux" | sort -k3 -nr | head -10
```

**🕵️ PHASE 3 : INVESTIGATION APPROFONDIE**

### 5. Recherche d'Artefacts Malveillants
```bash
# Fichiers modifiés récemment
find /var/log -type f -mtime -1 -ls
find /tmp -type f -mtime -1 -ls
find /home -type f -mtime -1 -ls

# Connexions réseau établies
lsof -i | grep ESTABLISHED

# Processus avec connexions réseau
lsof -i | grep "192.168.1.100"
```

### 6. Vérification de l'Intégrité Système
```bash
# Hash des binaires système critiques
for bin in /bin/ls /usr/bin/ssh /usr/sbin/sshd; do
    echo "$bin: $(sha256sum $bin)"
done

# Vérification des packages (Debian/Ubuntu)
debsums -c

# Vérification des packages (RHEL/CentOS)
rpm -Va
```

**🚨 PHASE 4 : DÉTECTION D'INTRUSION**

### 7. Analyse Mémoire (si compromission suspectée)
```bash
# Dump mémoire avec volatility (si possible)
sudo dd if=/dev/mem of=/tmp/memory_dump.raw bs=1M

# Analyse des connexions actives
volatility -f /tmp/memory_dump.raw --profile=LinuxUbuntu2004x64 linux_netstat

# Processus cachés
volatility -f /tmp/memory_dump.raw --profile=LinuxUbuntu2004x64 linux_pslist
```

### 8. Collecte d'Evidences
```bash
# Sauvegarde des logs critiques
tar -czf incident_logs_$(date +%Y%m%d_%H%M%S).tar.gz \
    /var/log/auth.log \
    /var/log/syslog \
    /var/log/apache2/ \
    /var/log/nginx/

# Capture réseau en cours
tcpdump -i any -w capture_$(date +%Y%m%d_%H%M%S).pcap host 192.168.1.100 &
```

**⚡ ACTIONS IMMÉDIATES DE CONFINEMENT**

### 9. Isolation et Blocage
```bash
# Bloquer l'IP au niveau firewall
iptables -I INPUT -s 192.168.1.100 -j DROP
iptables -I OUTPUT -d 192.168.1.100 -j DROP

# Sauvegarder les règles
iptables-save > /etc/iptables/rules.v4

# Tuer les connexions existantes
ss -K dst 192.168.1.100
```

### 10. Monitoring Continu
```bash
# Surveillance en temps réel
tail -f /var/log/auth.log | grep --color "192.168.1.100"

# Alerte sur nouvelles tentatives
while true; do
    if grep -q "192.168.1.100" /var/log/auth.log; then
        echo "ALERT: Nouvelle activité détectée de 192.168.1.100" | \
        mail -s "Security Alert" admin@company.com
    fi
    sleep 60
done
```

**📝 RAPPORT D'INCIDENT**

### Modèle de Rapport
```markdown
## RAPPORT D'INCIDENT DE SÉCURITÉ

**Date** : $(date)
**Analyste** : [Votre nom]
**Incident ID** : INC-$(date +%Y%m%d-%H%M%S)

### RÉSUMÉ EXÉCUTIF
- IP suspecte détectée : 192.168.1.100
- Première détection : [timestamp]
- Niveau de criticité : [FAIBLE/MOYEN/ÉLEVÉ/CRITIQUE]

### TIMELINE
- [Timestamp] : Première connexion détectée
- [Timestamp] : Actions malveillantes identifiées
- [Timestamp] : Mesures de confinement appliquées

### IMPACT
- Systèmes affectés : [liste]
- Données compromises : [OUI/NON/INCONNU]
- Services interrompus : [liste]

### ACTIONS ENTREPRISES
- Blocage IP source
- Isolation des systèmes affectés
- Collecte d'evidences
- Notification des parties prenantes

### RECOMMANDATIONS
- Renforcement des contrôles d'accès
- Mise à jour des règles de détection
- Formation du personnel
- Audit de sécurité complet
```

**🔄 PROCHAINES ÉTAPES**

1. **Forensics Approfondie** : Analyse des dumps mémoire et disque
2. **Attribution** : Identifier l'origine de l'attaque
3. **Remediation** : Correction des vulnérabilités exploitées
4. **Lessons Learned** : Documentation et amélioration des processus
5. **Monitoring Renforcé** : Surveillance continue des IoCs identifiés
```

## ⚙️ Configuration Avancée

### Fichiers de Configuration

#### Configuration Principale
```json
// ~/.llama-cybersec/config/local_config.json
{
  "model": {
    "default_model_id": "llama-cybersec-q4_0",
    "auto_download": true,
    "preferred_format": "gguf"
  },
  "inference": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "context_length": 2048
  },
  "web_ui": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8080,
    "theme": "dark"
  },
  "api": {
    "enabled": true,
    "port": 8081,
    "rate_limiting": {
      "requests_per_minute": 60
    }
  },
  "performance": {
    "enable_gpu": true,
    "cpu_threads": 0,
    "cache_size_mb": 512
  },
  "security": {
    "input_validation": true,
    "max_input_length": 4096,
    "content_filtering": true
  }
}
```

#### Profils Personnalisés
```json
// ~/.llama-cybersec/config/profiles/penetration_testing.json
{
  "name": "Pentest Assistant",
  "description": "Configuration optimisée pour les tests de pénétration",
  "inference": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 1024
  },
  "prompts": {
    "system_prompt": "Tu es un expert en tests de pénétration éthiques. Fournis des conseils techniques précis tout en rappelant l'importance de l'autorisation légale.",
    "examples": [
      "Expliquer les techniques de reconnaissance passive",
      "Analyser une vulnérabilité web",
      "Créer un payload personnalisé"
    ]
  }
}
```

### Variables d'Environnement

```bash
# Configuration système
export LLAMA_CYBERSEC_HOME="$HOME/.llama-cybersec"
export LLAMA_CYBERSEC_MODEL_PATH="$LLAMA_CYBERSEC_HOME/models"
export LLAMA_CYBERSEC_CONFIG_PATH="$LLAMA_CYBERSEC_HOME/config"

# Performance
export LLAMA_CYBERSEC_GPU_ENABLED="true"
export LLAMA_CYBERSEC_CPU_THREADS="0"  # Auto-detect
export LLAMA_CYBERSEC_CACHE_SIZE="512"  # MB

# Sécurité
export LLAMA_CYBERSEC_MAX_INPUT_LENGTH="4096"
export LLAMA_CYBERSEC_ENABLE_CONTENT_FILTER="true"

# Logging
export LLAMA_CYBERSEC_LOG_LEVEL="INFO"
export LLAMA_CYBERSEC_LOG_REQUESTS="false"

# API
export LLAMA_CYBERSEC_API_HOST="127.0.0.1"
export LLAMA_CYBERSEC_API_PORT="8081"
```

### Configuration GPU

#### NVIDIA GPU
```json
{
  "gpu_config": {
    "provider": "cuda",
    "device_id": 0,
    "memory_fraction": 0.8,
    "allow_growth": true,
    "precision": "fp16"
  }
}
```

#### AMD GPU (ROCm)
```json
{
  "gpu_config": {
    "provider": "rocm", 
    "device_id": 0,
    "memory_fraction": 0.8
  }
}
```

#### Intel GPU (OneAPI)
```json
{
  "gpu_config": {
    "provider": "intel",
    "device_id": 0,
    "backend": "level_zero"
  }
}
```

### Optimisations Avancées

#### Modèles Quantifiés
```bash
# Q4_0 : Équilibre qualité/vitesse (recommandé)
python utils/model_manager.py --install llama-cybersec-q4_0

# Q8_0 : Haute qualité, plus lent
python utils/model_manager.py --install llama-cybersec-q8_0

# Q2_K : Ultra rapide, qualité réduite
python utils/model_manager.py --install llama-cybersec-q2_k
```

#### Cache Intelligent
```json
{
  "cache_config": {
    "enable_kv_cache": true,
    "cache_size_mb": 1024,
    "cache_strategy": "lru",
    "persist_cache": false
  }
}
```

## 🔧 Dépannage

### Problèmes Fréquents

#### 1. Modèle ne se charge pas

**Symptômes :**
- Interface web affiche "Modèle en cours de chargement"
- API retourne erreur 503

**Solutions :**
```bash
# Vérifier l'intégrité du modèle
python utils/model_manager.py --action info --model-id "votre-modele"

# Vérifier les logs
tail -f ~/.llama-cybersec/logs/interface_*.log

# Réinstaller le modèle
python utils/model_manager.py --action install --model-id "votre-modele" --force

# Vérifier l'espace disque
df -h ~/.llama-cybersec/

# Vérifier la mémoire disponible
free -h
```

#### 2. Erreur de mémoire

**Symptômes :**
- `OutOfMemoryError`
- Système devient lent
- Process killed

**Solutions :**
```json
// Réduire la taille du contexte
{
  "inference": {
    "context_length": 1024,  // Au lieu de 2048
    "batch_size": 1
  }
}
```

```bash
# Utiliser un modèle plus léger
python utils/model_manager.py --install llama-cybersec-q4_0

# Désactiver le GPU si problème VRAM
export LLAMA_CYBERSEC_GPU_ENABLED="false"
```

#### 3. Interface web inaccessible

**Symptômes :**
- Connexion refusée sur port 8080
- Timeout de connexion

**Solutions :**
```bash
# Vérifier que le service fonctionne
ps aux | grep local_interface

# Vérifier le port
netstat -an | grep 8080

# Vérifier les logs
tail -f ~/.llama-cybersec/logs/interface_*.log

# Changer le port
python local_deployment/local_interface.py --port 8081

# Vérifier le firewall
sudo ufw status
sudo ufw allow 8080
```

#### 4. Réponses de mauvaise qualité

**Symptômes :**
- Réponses incohérentes
- Répétitions excessives
- Hors sujet

**Solutions :**
```json
// Ajuster les paramètres
{
  "inference": {
    "temperature": 0.7,     // Réduire pour plus de cohérence
    "top_p": 0.9,          // Réduire pour moins de créativité
    "repetition_penalty": 1.1,
    "max_tokens": 512
  }
}
```

```bash
# Vérifier la version du modèle
python utils/model_manager.py --action info --model-id "votre-modele"

# Mettre à jour vers une version plus récente
python utils/model_manager.py --action update --model-id "votre-modele"
```

#### 5. Performance lente

**Symptômes :**
- Réponses prennent plus de 10 secondes
- CPU/GPU à 100%

**Solutions :**
```bash
# Vérifier les ressources système
htop
nvidia-smi  # Si GPU NVIDIA

# Optimiser les threads CPU
export LLAMA_CYBERSEC_CPU_THREADS="8"  # Ajuster selon votre CPU

# Activer le cache
```

```json
{
  "performance": {
    "enable_gpu": true,
    "cache_size_mb": 1024,
    "preload_model": true,
    "use_mmap": true
  }
}
```

### Diagnostic Avancé

#### Collecte d'Informations Système
```bash
#!/bin/bash
# Script de diagnostic

echo "=== LLaMA CyberSec Diagnostic ==="
echo "Date: $(date)"
echo

echo "=== Système ==="
uname -a
lsb_release -a 2>/dev/null || cat /etc/os-release
echo

echo "=== Python ==="
python3 --version
pip list | grep -E "(torch|transformers|flask)"
echo

echo "=== Mémoire ==="
free -h
echo

echo "=== GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "NVIDIA GPU non détecté"
fi
echo

echo "=== Modèles Installés ==="
python utils/model_manager.py --action list
echo

echo "=== Configuration ==="
if [ -f ~/.llama-cybersec/config/local_config.json ]; then
    cat ~/.llama-cybersec/config/local_config.json
else
    echo "Configuration non trouvée"
fi
echo

echo "=== Logs Récents ==="
tail -20 ~/.llama-cybersec/logs/interface_*.log 2>/dev/null || echo "Pas de logs trouvés"
```

#### Test de Performance
```python
#!/usr/bin/env python3
# Test de performance
import time
import requests
import statistics

def benchmark_api():
    url = "http://localhost:8081/api/generate"
    test_prompt = "Expliquer rapidement ce qu'est une attaque par déni de service"
    
    times = []
    for i in range(10):
        start = time.time()
        
        response = requests.post(url, json={
            "prompt": test_prompt,
            "max_tokens": 100,
            "temperature": 0.7
        })
        
        end = time.time()
        
        if response.status_code == 200:
            times.append(end - start)
            print(f"Test {i+1}: {end - start:.2f}s")
        else:
            print(f"Test {i+1}: ERREUR {response.status_code}")
    
    if times:
        print(f"\nRésultats:")
        print(f"Temps moyen: {statistics.mean(times):.2f}s")
        print(f"Temps médian: {statistics.median(times):.2f}s")
        print(f"Min: {min(times):.2f}s")
        print(f"Max: {max(times):.2f}s")

if __name__ == "__main__":
    benchmark_api()
```

### Support et Communauté

#### Ressources d'Aide

1. **Documentation** : `docs/` dans le repository
2. **Issues GitHub** : Signaler des bugs ou demander des fonctionnalités
3. **Discussions** : Forum communautaire pour questions et échanges
4. **Wiki** : Base de connaissances collaborative

#### Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche pour votre fonctionnalité
3. Commit et push vos changements  
4. Ouvrir une Pull Request

#### Contact

- **Email** : support@llama-cybersec.com
- **Discord** : Serveur communautaire LLaMA CyberSec
- **Twitter** : @LLaMACyberSec

---

Ce guide couvre l'utilisation complète du système LLaMA CyberSec v2.0. Pour des questions spécifiques ou des problèmes non couverts, consultez la documentation technique ou contactez la communauté.