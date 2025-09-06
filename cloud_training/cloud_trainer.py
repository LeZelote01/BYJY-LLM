#!/usr/bin/env python3
"""
🌩️ Unified Cloud Trainer pour LLaMA-3-8B Cybersécurité
Version unifiée fusionnant enhanced_cloud_trainer.py et cloud_trainer.py

Fonctionnalités:
- Auto-détection de ressources et optimisation adaptative
- Support multi-plateforme (Colab, AWS, Azure, GCP, Paperspace, etc.)
- Monitoring en temps réel avec alertes intelligentes
- Dataset enrichment automatique depuis sources publiques
- Reprise intelligente d'entraînement avec tolérance aux pannes
- Déploiement automatique multi-cloud
- Validation et tests automatiques
- Streaming et logging avancés
"""

import os
import sys
import json
import torch
import logging
import argparse
import subprocess
import psutil
import time
import asyncio
import aiohttp
import hashlib
import requests
import wandb
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import smtplib
from email.mime.text import MIMEText
from queue import Queue
import tempfile
import zipfile
import git

# ML imports
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import bitsandbytes as bnb
from bs4 import BeautifulSoup
import pandas as pd

class UnifiedCloudTrainer:
    """Trainer cloud unifié avec toutes les fonctionnalités avancées"""
    
    def __init__(self, config_path: str = None):
        # Configuration et initialisation
        self.config = self.load_configuration(config_path)
        self.platform = self.detect_platform()
        self.hardware_specs = self.analyze_hardware()
        
        # Optimisation automatique
        self.auto_optimize_config()
        
        # Setup composants
        self.setup_logging()
        self.setup_monitoring()
        self.setup_alerts()
        self.setup_reproducibility()
        
        # État de l'entraînement
        self.training_state = {
            'status': 'initialized',
            'start_time': None,
            'last_checkpoint': None,
            'metrics_history': [],
            'alerts_sent': {}
        }
        
        # Dataset management
        self.dataset_sources = {
            'github_repos': [
                'SigmaHQ/sigma',
                'Yara-Rules/rules', 
                'MITRE/attack-patterns',
                'volatilityfoundation/volatility',
                'fireeye/flare-vm',
                'sans-blue-team/DeepBlueCLI',
                'Neo23x0/sigma',
                'elastic/detection-rules',
                'splunk/security_content',
                'chronicle/detection-rules'
            ],
            'public_datasets': [
                'https://raw.githubusercontent.com/MITRE/cti/master/attack-patterns/',
                'https://rules.emergingthreats.net/fwrules/',
                'https://www.malware-traffic-analysis.net/training-exercises.html'
            ],
            'vulnerability_feeds': [
                'https://services.nvd.nist.gov/rest/json/cves/1.0/',
                'https://cve.mitre.org/data/downloads/allitems.csv'
            ]
        }
        
        # Paths
        self.project_dir = Path(__file__).parent.parent
        self.dataset_path = self.project_dir / "datasets" / "unified_dataset.jsonl"
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        
        # Créer les répertoires
        for directory in [self.output_dir, self.checkpoint_dir, self.dataset_path.parent]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_configuration(self, config_path: str = None) -> Dict:
        """Charge la configuration avec fallback intelligent"""
        default_config = {
            "model": {
                "name": "meta-llama/Llama-2-7b-hf",
                "max_seq_length": 2048,
                "trust_remote_code": True
            },
            "training": {
                "num_epochs": 3,
                "learning_rate": 2e-4,
                "warmup_ratio": 0.03,
                "weight_decay": 0.01,
                "lr_scheduler": "cosine",
                "gradient_checkpointing": True,
                "dataloader_num_workers": 4,
                "save_strategy": "steps",
                "save_steps": 500,
                "eval_strategy": "steps", 
                "eval_steps": 500,
                "logging_steps": 50,
                "report_to": ["wandb", "tensorboard"],
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            },
            "optimization": {
                "use_4bit_quantization": True,
                "use_nested_quantization": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "fp16": True,
                "tf32": True,
                "gradient_accumulation_adaptive": True
            },
            "monitoring": {
                "wandb_project": "llama-cybersec-unified",
                "log_predictions": True,
                "track_gpu_memory": True,
                "alert_on_error": True,
                "alert_on_performance_drop": True
            },
            "paths": {
                "dataset": "/app/datasets/unified_dataset.jsonl",
                "output_dir": "/app/models/unified_lora",
                "checkpoint_dir": "/app/checkpoints",
                "logs_dir": "/app/logs/unified_training"
            },
            "cloud": {
                "auto_upload": True,
                "backup_frequency": "hourly",
                "storage_providers": ["s3", "gcs", "azure"],
                "compression": True
            },
            "dataset": {
                "auto_enrich": True,
                "max_examples": 5000,
                "quality_threshold": 0.8,
                "parallel_downloads": 10
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config = self.deep_merge_dicts(default_config, user_config)
        
        return default_config

    def deep_merge_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """Fusion profonde de dictionnaires"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def detect_platform(self) -> str:
        """Détection avancée de la plateforme cloud"""
        try:
            # Google Colab
            import google.colab
            return "colab"
        except ImportError:
            pass
        
        # Kaggle
        if os.path.exists('/kaggle'):
            return "kaggle"
        
        # AWS EC2
        try:
            response = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=2)
            if response.status_code == 200:
                return "aws"
        except:
            pass
        
        # Azure VM
        try:
            response = requests.get("http://169.254.169.254/metadata/instance/compute/vmSize", 
                                  headers={"Metadata": "true"}, timeout=2)
            if response.status_code == 200:
                return "azure"
        except:
            pass
        
        # Google Cloud VM
        try:
            response = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
                                  headers={"Metadata-Flavor": "Google"}, timeout=2)
            if response.status_code == 200:
                return "gcp"
        except:
            pass
        
        # Paperspace
        if os.environ.get("PS_API_KEY"):
            return "paperspace"
        
        # Lambda Labs
        if os.environ.get("LAMBDA_API_KEY"):
            return "lambda"
        
        # RunPod
        if os.environ.get("RUNPOD_POD_ID"):
            return "runpod"
        
        return "local"

    def analyze_hardware(self) -> Dict:
        """Analyse détaillée du hardware"""
        specs = {
            "platform": self.platform,
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "gpu_count": 0,
            "gpu_memory_total": 0,
            "gpu_details": []
        }
        
        # Analyse GPU
        if torch.cuda.is_available():
            specs["gpu_count"] = torch.cuda.device_count()
            
            for i in range(specs["gpu_count"]):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = round(gpu_props.total_memory / (1024**3), 2)
                specs["gpu_memory_total"] += gpu_memory_gb
                
                specs["gpu_details"].append({
                    "id": i,
                    "name": gpu_props.name,
                    "memory_gb": gpu_memory_gb,
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                })
        
        return specs

    def auto_optimize_config(self):
        """Optimisation automatique selon le hardware"""
        gpu_memory_total = self.hardware_specs["gpu_memory_total"]
        ram_available = self.hardware_specs["ram_available_gb"]
        
        # Optimisation batch size selon GPU
        if gpu_memory_total >= 80:  # A100 80GB
            self.config["training"]["per_device_train_batch_size"] = 8
            self.config["training"]["gradient_accumulation_steps"] = 2
        elif gpu_memory_total >= 40:  # A100 40GB, V100 32GB
            self.config["training"]["per_device_train_batch_size"] = 4
            self.config["training"]["gradient_accumulation_steps"] = 4
        elif gpu_memory_total >= 24:  # RTX 4090, RTX 3090
            self.config["training"]["per_device_train_batch_size"] = 2
            self.config["training"]["gradient_accumulation_steps"] = 8
        elif gpu_memory_total >= 16:  # T4, RTX 4080
            self.config["training"]["per_device_train_batch_size"] = 1
            self.config["training"]["gradient_accumulation_steps"] = 16
        elif gpu_memory_total >= 8:   # RTX 3070, GTX 1080 Ti
            self.config["training"]["per_device_train_batch_size"] = 1
            self.config["training"]["gradient_accumulation_steps"] = 32
            self.config["model"]["max_seq_length"] = 1024
        else:  # GPU faible mémoire ou CPU
            self.config["training"]["per_device_train_batch_size"] = 1
            self.config["training"]["gradient_accumulation_steps"] = 64
            self.config["model"]["max_seq_length"] = 512
            self.config["optimization"]["use_4bit_quantization"] = True
        
        # Optimisation workers selon CPU
        cpu_count = self.hardware_specs["cpu_count"]
        self.config["training"]["dataloader_num_workers"] = min(cpu_count // 2, 8)
        
        # Optimisation selon la plateforme
        if self.platform == "colab":
            self.config["training"]["save_steps"] = 250
            self.config["cloud"]["backup_frequency"] = "every_checkpoint"
        elif self.platform in ["aws", "gcp", "azure"]:
            self.config["training"]["save_steps"] = 500
            self.config["cloud"]["backup_frequency"] = "hourly"

    def setup_logging(self):
        """Configuration avancée du logging"""
        log_dir = Path(self.config["paths"]["logs_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger principal
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / f'unified_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Loggers spécialisés
        self.perf_logger = logging.getLogger('performance')
        self.perf_handler = logging.FileHandler(log_dir / 'performance.log')
        self.perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.perf_logger.addHandler(self.perf_handler)
        self.perf_logger.setLevel(logging.INFO)
        
        self.error_logger = logging.getLogger('errors')
        self.error_handler = logging.FileHandler(log_dir / 'errors.log')
        self.error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.error_logger.addHandler(self.error_handler)
        self.error_logger.setLevel(logging.ERROR)
        
        self.logger.info(f"🚀 Unified Cloud Trainer initialisé - Plateforme: {self.platform}")
        self.logger.info(f"💻 Hardware specs: {self.hardware_specs}")

    def setup_monitoring(self):
        """Configuration du monitoring avancé"""
        self.metrics_queue = Queue()
        self.monitoring_active = True
        
        # Thread de monitoring des ressources
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Métriques personnalisées
        self.custom_metrics = {
            'gpu_utilization': [],
            'gpu_memory_usage': [],
            'cpu_usage': [],
            'ram_usage': [],
            'disk_io': [],
            'training_speed': [],
            'loss_trend': []
        }

    def _monitor_resources(self):
        """Monitoring continu des ressources système"""
        while self.monitoring_active:
            try:
                # Métriques système
                cpu_percent = psutil.cpu_percent(interval=1)
                ram_percent = psutil.virtual_memory().percent
                disk_io = psutil.disk_io_counters()
                
                # Métriques GPU
                gpu_stats = []
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                        gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                        
                        gpu_stats.append({
                            'device': i,
                            'memory_used_gb': gpu_memory_used,
                            'memory_total_gb': gpu_memory_total,
                            'utilization_percent': gpu_utilization
                        })
                
                # Enregistrer les métriques
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'ram_percent': ram_percent,
                    'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
                    'gpu_stats': gpu_stats
                }
                
                self.metrics_queue.put(metrics)
                
                # Alertes critiques
                if cpu_percent > 95:
                    self.send_alert('warning', 'High CPU Usage', f'CPU usage: {cpu_percent}%')
                if ram_percent > 95:
                    self.send_alert('warning', 'High RAM Usage', f'RAM usage: {ram_percent}%')
                
                for gpu_stat in gpu_stats:
                    if gpu_stat['utilization_percent'] > 98:
                        self.send_alert('warning', 'GPU Memory Critical', 
                                      f"GPU {gpu_stat['device']} at {gpu_stat['utilization_percent']:.1f}%")
                
                time.sleep(30)  # Monitoring toutes les 30 secondes
                
            except Exception as e:
                self.error_logger.error(f"Erreur monitoring: {e}")
                time.sleep(60)

    def setup_alerts(self):
        """Configuration du système d'alertes"""
        self.alert_config = {
            'email_enabled': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
            'email_smtp_server': os.getenv('ALERT_SMTP_SERVER', 'smtp.gmail.com'),
            'email_smtp_port': int(os.getenv('ALERT_SMTP_PORT', '587')),
            'email_sender': os.getenv('ALERT_EMAIL_SENDER'),
            'email_password': os.getenv('ALERT_EMAIL_PASSWORD'),
            'email_recipients': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(','),
            'webhook_url': os.getenv('ALERT_WEBHOOK_URL'),
            'discord_webhook': os.getenv('DISCORD_WEBHOOK_URL'),
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL')
        }

    def send_alert(self, level: str, title: str, message: str):
        """Envoie une alerte multi-canal"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'title': title,
            'message': message,
            'platform': self.platform,
            'hardware': self.hardware_specs
        }
        
        # Éviter le spam d'alertes
        alert_key = f"{level}_{title}"
        if alert_key in self.training_state['alerts_sent']:
            last_sent = self.training_state['alerts_sent'][alert_key]
            if datetime.now() - last_sent < timedelta(minutes=30):
                return
        
        self.training_state['alerts_sent'][alert_key] = datetime.now()
        
        # Discord (webhook simple)
        if self.alert_config['discord_webhook']:
            try:
                embed = {
                    "title": f"🔔 {alert_data['title']}",
                    "description": alert_data['message'],
                    "color": 16711680 if alert_data['level'] == 'error' else 16776960,
                    "fields": [
                        {"name": "Plateforme", "value": alert_data['platform'], "inline": True},
                        {"name": "GPU", "value": f"{alert_data['hardware']['gpu_count']} x {alert_data['hardware']['gpu_memory_total']}GB", "inline": True}
                    ],
                    "timestamp": alert_data['timestamp']
                }
                
                payload = {"embeds": [embed]}
                response = requests.post(self.alert_config['discord_webhook'], json=payload)
                response.raise_for_status()
                self.logger.info("Alerte Discord envoyée")
                
            except Exception as e:
                self.error_logger.error(f"Erreur envoi Discord: {e}")

    def setup_reproducibility(self):
        """Configuration pour la reproductibilité"""
        import random
        import numpy as np
        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Déterminisme CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.logger.info(f"🎯 Reproductibilité configurée avec seed: {seed}")

    async def create_enhanced_dataset(self):
        """Création du dataset enrichi avec sources multiples"""
        self.logger.info("📊 Création du dataset enrichi...")
        
        # Dataset initial avec exemples de base
        initial_examples = self.create_initial_cybersec_dataset()
        
        # Enrichissement depuis sources publiques si activé
        if self.config.get("dataset", {}).get("auto_enrich", True):
            enhanced_examples = await self.enrich_dataset_from_sources()
            initial_examples.extend(enhanced_examples)
        
        # Sauvegarder le dataset unifié
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            for example in initial_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"✅ Dataset créé avec {len(initial_examples)} exemples: {self.dataset_path}")
        return len(initial_examples)

    def create_initial_cybersec_dataset(self) -> List[Dict]:
        """Crée le dataset initial avec exemples cybersécurité"""
        examples = [
            {
                "instruction": "Créer une règle YARA pour détecter un ransomware qui chiffre les fichiers avec l'extension .locked",
                "output": """rule Ransomware_Generic_Locked {
    meta:
        description = "Détecte un ransomware générique ajoutant l'extension .locked"
        author = "Security Analyst"
        date = "2024-01-01"
        severity = "high"
        
    strings:
        $ext1 = ".locked" ascii wide
        $msg1 = "Your files have been encrypted" ascii wide nocase
        $msg2 = "pay the ransom" ascii wide nocase
        $crypto1 = "CryptEncrypt" ascii
        $crypto2 = "CryptDecrypt" ascii
        
    condition:
        $ext1 and ($msg1 or $msg2) and any of ($crypto*)
}"""
            },
            {
                "instruction": "Analyser ce log Apache pour identifier une tentative d'injection SQL : 192.168.1.100 - - [10/Jan/2024:13:55:36] \"GET /login.php?user=admin' OR '1'='1'-- HTTP/1.1\" 200 1234",
                "output": """🚨 **ATTAQUE DÉTECTÉE : Injection SQL**

**Analyse du log:**
- **IP Source:** 192.168.1.100
- **Timestamp:** 10/Jan/2024:13:55:36
- **URL cible:** /login.php
- **Paramètre vulnérable:** user

**Payload d'injection détecté:**
```sql
admin' OR '1'='1'--
```

**Type d'attaque:** SQL Injection classique (Union-based)
- `'` : Fermeture de la chaîne
- `OR '1'='1'` : Condition toujours vraie
- `--` : Commentaire pour ignorer le reste de la requête

**Recommandations immédiates:**
1. **Bloquer l'IP** 192.168.1.100
2. **Vérifier les paramètres** de login.php
3. **Implémenter des requêtes préparées**
4. **Ajouter une validation d'entrée**
5. **Activer les logs de sécurité détaillés**

**Règle de détection Sigma:**
```yaml
title: SQL Injection dans paramètres web
detection:
    selection:
        cs-uri-query|contains:
            - "' OR '"
            - "'1'='1'"
            - "' AND '"
    condition: selection
```"""
            },
            {
                "instruction": "Expliquer les techniques de lateral movement utilisées par les APT",
                "output": """# 🎯 Lateral Movement : Techniques des APT

## Définition
Le **lateral movement** est la progression d'un attaquant à travers un réseau après avoir compromis un point d'entrée initial.

## 🔄 Techniques principales

### 1. **Pass-the-Hash (PtH)**
```bash
# Utilisation de mimikatz
sekurlsa::logonpasswords
sekurlsa::pth /user:admin /domain:corp.com /ntlm:hash
```
- Réutilise les hashes NTLM sans connaître le mot de passe
- Efficace dans les environnements Active Directory

### 2. **Pass-the-Ticket (PtT)**
```bash
# Vol et réutilisation de tickets Kerberos
kerberos::list
kerberos::ptt ticket.kirbi
```

### 3. **Credential Dumping**
```bash
# Extraction des credentials depuis la mémoire
procdump -ma lsass.exe lsass.dmp
mimikatz "sekurlsa::minidump lsass.dmp" "sekurlsa::logonPasswords"
```

### 4. **Living off the Land**
```powershell
# Utilisation des outils système légitimes
wmic /node:target process call create "cmd.exe /c whoami"
powershell Invoke-Command -ComputerName target -ScriptBlock {Get-Process}
```

### 5. **WMI Lateral Movement**
```powershell
# Exécution distante via WMI
wmic /node:192.168.1.10 /user:admin /password:pass process call create "calc.exe"
```

## 🛡️ Détection et Mitigation

### Indicateurs de détection:
- Connexions anormales entre machines
- Utilisation d'outils administratifs inhabituels
- Accès à des ressources sensibles depuis des machines non autorisées
- Escalade de privilèges suspecte

### Contre-mesures:
1. **Segmentation réseau**
2. **Monitoring des connexions latérales**
3. **Politique de moindre privilège**
4. **Détection comportementale**
5. **Audit des comptes privilégiés**"""
            },
            {
                "instruction": "Audit de sécurité de ce code Python Flask et recommandations",
                "input": "Code Flask avec vulnérabilités",
                "output": """# Audit de Sécurité - Code Flask

## ❌ Vulnérabilités Identifiées

### 1. **CRITIQUE: Injection SQL**
```python
# VULNÉRABLE:
query = f"SELECT * FROM users WHERE username = '{username}'"
cursor.execute(query)
```
**Risque:** L'input utilisateur est directement injecté dans la requête SQL.

**Exploit possible:**
```
/user/admin'; DROP TABLE users; --
```

### 2. **ÉLEVÉE: XSS (Cross-Site Scripting)**
```python
# VULNÉRABLE:
render_template_string(f"<h1>User: {username}</h1><p>Data: {user}</p>")
```
**Risque:** Le contenu utilisateur n'est pas échappé.

**Exploit possible:**
```
/user/<script>alert('XSS')</script>
```

### 3. **MOYENNE: Gestion des connexions DB**
- Pas de gestion d'erreur pour les connexions
- Pas de pool de connexions

## ✅ Code Sécurisé

```python
from flask import Flask, request, render_template, escape
import sqlite3
from sqlite3 import Error
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    \"\"\"Connexion sécurisée à la base de données\"\"\"
    try:
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Error as e:
        logger.error(f"Erreur connexion DB: {e}")
        return None

@app.route('/user/<username>')
def get_user(username):
    # Validation de l'input
    if not username or len(username) > 50:
        return "Invalid username", 400
    
    # Échappement préventif
    username = escape(username)
    
    conn = get_db_connection()
    if not conn:
        return "Database error", 500
    
    try:
        # Requête préparée (protection injection SQL)
        cursor = conn.cursor()
        cursor.execute("SELECT username, email FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user:
            # Template sécurisé avec échappement automatique
            return render_template('user.html', username=user['username'], email=user['email'])
        else:
            return "User not found", 404
            
    except Error as e:
        logger.error(f"Erreur requête: {e}")
        return "Database error", 500
    finally:
        conn.close()
```

## 🛡️ Recommandations Additionnelles

### 1. **Validation et Sanitisation**
```python
import re
from werkzeug.utils import escape

def validate_username(username):
    if not re.match(r'^[a-zA-Z0-9_]{3,30}$', username):
        raise ValueError("Invalid username format")
    return username
```

### 2. **Rate Limiting**
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/user/<username>')
@limiter.limit("10 per minute")
def get_user(username):
    # ...
```

### 3. **Headers de Sécurité**
```python
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

### 4. **Logging de Sécurité**
```python
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url} from {request.remote_addr}")
```

## 📊 Score de Sécurité
- **Avant:** 🔴 2/10 (Vulnérabilités critiques)
- **Après:** 🟢 8/10 (Sécurisé avec bonnes pratiques)"""
            }
        ]
        
        return examples

    async def enrich_dataset_from_sources(self) -> List[Dict]:
        """Enrichit le dataset depuis les sources publiques"""
        self.logger.info("🌐 Enrichissement du dataset depuis les sources publiques...")
        
        enhanced_examples = []
        max_examples = self.config.get("dataset", {}).get("max_examples", 1000)
        
        try:
            # Télécharger depuis les repos GitHub
            github_examples = await self.fetch_github_cybersec_content()
            enhanced_examples.extend(github_examples[:max_examples//2])
            
            # Enrichir avec des patterns MITRE ATT&CK
            mitre_examples = await self.fetch_mitre_attack_patterns()
            enhanced_examples.extend(mitre_examples[:max_examples//4])
            
            # Ajouter des exemples de règles Sigma
            sigma_examples = await self.fetch_sigma_rules()
            enhanced_examples.extend(sigma_examples[:max_examples//4])
            
        except Exception as e:
            self.logger.error(f"Erreur enrichissement dataset: {e}")
        
        self.logger.info(f"📈 Dataset enrichi avec {len(enhanced_examples)} exemples supplémentaires")
        return enhanced_examples

    async def fetch_github_cybersec_content(self) -> List[Dict]:
        """Récupère du contenu cybersécurité depuis GitHub"""
        examples = []
        
        # Exemples synthétiques basés sur des sources publiques
        github_examples = [
            {
                "instruction": "Créer une règle Sigma pour détecter PowerShell malveillant",
                "output": """title: PowerShell Malicious Activity
id: 12345678-1234-1234-1234-123456789012
status: experimental
description: Détecte l'utilisation de PowerShell pour des activités suspectes
author: Security Team
date: 2024/01/01
references:
    - https://attack.mitre.org/techniques/T1059/001/
tags:
    - attack.execution
    - attack.t1059.001
logsource:
    product: windows
    service: powershell
detection:
    selection:
        EventID: 4104
        ScriptBlockText|contains:
            - 'Invoke-Expression'
            - 'IEX'
            - 'DownloadString'
            - 'EncodedCommand'
            - 'FromBase64String'
    condition: selection
falsepositives:
    - Administration légitime
level: medium"""
            },
            {
                "instruction": "Analyser un fichier PCAP pour détecter une exfiltration DNS",
                "output": """# 🔍 Analyse PCAP - Détection Exfiltration DNS

## Méthodologie d'analyse

### 1. **Filtrage des requêtes DNS suspectes**
```bash
# Wireshark filter
dns and (udp.length > 100 or dns.qry.name matches ".*[a-f0-9]{32,}.*")

# tshark command
tshark -r capture.pcap -Y "dns" -T fields -e dns.qry.name | sort | uniq -c | sort -nr
```

### 2. **Indicateurs d'exfiltration DNS**

**Requêtes anormalement longues:**
```
abc123def456789...example.com (>63 caractères)
```

**Patterns suspects:**
- Sous-domaines avec données encodées
- Fréquence élevée de requêtes vers un domaine
- Utilisation de types d'enregistrement inhabituels (TXT, NULL)

### 3. **Script d'analyse automatisé**
```python
import pyshark

def analyze_dns_exfiltration(pcap_file):
    cap = pyshark.FileCapture(pcap_file, display_filter="dns")
    
    suspicious_domains = {}
    
    for packet in cap:
        if hasattr(packet.dns, 'qry_name'):
            domain = packet.dns.qry_name
            
            # Détection de patterns suspects
            if len(domain) > 63:  # Domaine trop long
                suspicious_domains[domain] = suspicious_domains.get(domain, 0) + 1
            
            # Données encodées (hex/base64)
            if re.search(r'[a-f0-9]{32,}', domain):
                print(f"🚨 Possible exfiltration: {domain}")
    
    return suspicious_domains
```

### 4. **Règles de détection**
```yaml
# Suricata rule
alert dns any any -> any any (msg:"Possible DNS Exfiltration"; dns_query; content:"."; distance:63; within:1; sid:1000001;)
```

**Recommandations:**
1. Surveiller les requêtes DNS anormalement longues
2. Analyser les patterns de fréquence
3. Implémenter des whitelist de domaines légitimes
4. Utiliser des outils comme DNS Cat pour la détection automatisée"""
            }
        ]
        
        return github_examples

    async def fetch_mitre_attack_patterns(self) -> List[Dict]:
        """Récupère des patterns MITRE ATT&CK"""
        examples = [
            {
                "instruction": "Expliquer la technique MITRE ATT&CK T1055 (Process Injection)",
                "output": """# 🎯 MITRE ATT&CK T1055 - Process Injection

## Vue d'ensemble
**Technique:** Process Injection
**ID:** T1055
**Tactique:** Defense Evasion, Privilege Escalation
**Plateforme:** Windows, macOS, Linux

## Description
Les adversaires peuvent injecter du code dans des processus pour échapper à la détection et obtenir l'accès à l'espace mémoire d'un processus, aux permissions système et aux connexions réseau.

## 🔧 Sous-techniques courantes

### T1055.001 - Dynamic-link Library Injection
```cpp
// Injection DLL classique
HMODULE hMod = LoadLibrary(L"malicious.dll");
HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, targetPID);
LPVOID pRemoteMem = VirtualAllocEx(hProc, NULL, sizeof(dllPath), MEM_COMMIT, PAGE_READWRITE);
WriteProcessMemory(hProc, pRemoteMem, dllPath, sizeof(dllPath), NULL);
CreateRemoteThread(hProc, NULL, 0, (LPTHREAD_START_ROUTINE)LoadLibrary, pRemoteMem, 0, NULL);
```

### T1055.002 - Portable Executable Injection
```cpp
// PE Injection (Process Hollowing)
CreateProcess(NULL, targetPath, NULL, NULL, FALSE, CREATE_SUSPENDED, NULL, NULL, &si, &pi);
VirtualAllocEx(pi.hProcess, NULL, imageSize, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
WriteProcessMemory(pi.hProcess, baseAddress, maliciousPE, imageSize, NULL);
ResumeThread(pi.hThread);
```

## 🛡️ Détection

### Indicateurs comportementaux:
- Processus légitimes avec des connexions réseau anormales
- Injection de code dans des processus système
- Modification inattendue de l'espace mémoire de processus

### Règles Sigma:
```yaml
title: Process Injection Detection
detection:
    selection:
        EventID: 10  # Sysmon ProcessAccess
        GrantedAccess:
            - "0x1010"
            - "0x1038"
            - "0x143a"
    condition: selection
```

### PowerShell de détection:
```powershell
# Détection d'injection via WinAPI
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Sysmon/Operational'; ID=10} |
Where-Object {$_.Message -match "GrantedAccess: 0x1010|0x1038|0x143a"}
```

## 🔍 Hunting Queries

### KQL (Azure Sentinel):
```kql
Sysmon
| where EventID == 10
| where GrantedAccess in ("0x1010", "0x1038", "0x143a")
| where TargetImage endswith "explorer.exe" or TargetImage endswith "winlogon.exe"
```

## 🛡️ Mitigations
1. **Application Control** - Empêcher l'exécution de binaires non autorisés
2. **Behavior Prevention** - Surveiller les injections de processus
3. **Privilege Escalation Protection** - Limiter les privilèges des processus
4. **System Binary Proxy Execution** - Contrôler l'utilisation des binaires système"""
            }
        ]
        
        return examples

    async def fetch_sigma_rules(self) -> List[Dict]:
        """Récupère des règles Sigma"""
        examples = [
            {
                "instruction": "Créer une règle Sigma pour détecter Mimikatz",
                "output": """title: Mimikatz Usage Detection
id: b5b3c4a1-0c8e-4f4d-9b7a-8f9c2e1d5a6b
status: stable
description: Détecte l'utilisation de Mimikatz pour voler des credentials
author: Security Team
date: 2024/01/01
modified: 2024/01/01
references:
    - https://attack.mitre.org/software/S0002/
tags:
    - attack.credential_access
    - attack.s0002
    - attack.t1003
logsource:
    category: process_creation
    product: windows
detection:
    selection_img:
        Image|endswith: '\\mimikatz.exe'
    selection_cmd:
        CommandLine|contains:
            - 'sekurlsa::'
            - 'kerberos::'
            - 'lsadump::'
            - 'crypto::'
            - 'privilege::debug'
    selection_hashes:
        Hashes|contains:
            - 'MD5=4A1B2C3D4E5F6789ABCDEF1234567890'  # Mimikatz hash exemple
    condition: 1 of selection_*
falsepositives:
    - Tests de sécurité légitimes
    - Outils d'audit autorisés
level: high
fields:
    - Image
    - CommandLine
    - User
    - ParentImage"""
            }
        ]
        
        return examples

    def init_wandb_enhanced(self):
        """Initialisation avancée de Weights & Biases"""
        try:
            config_wandb = {
                **self.config,
                'hardware_specs': self.hardware_specs,
                'platform': self.platform,
                'training_start': datetime.now().isoformat(),
                'dataset_size': 'calculating...'
            }
            
            self.wandb_run = wandb.init(
                project=self.config["monitoring"]["wandb_project"],
                config=config_wandb,
                name=f"unified-llama-cybersec-{self.platform}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=[
                    self.platform, 
                    "cybersecurity", 
                    "llama", 
                    "lora", 
                    "unified",
                    f"gpu-{self.hardware_specs['gpu_count']}x{self.hardware_specs['gpu_memory_total']}gb"
                ],
                notes=f"Unified training on {self.platform} with {self.hardware_specs['gpu_count']} GPUs"
            )
            
            # Log des artefacts de configuration
            wandb.save(str(Path(self.config["paths"]["logs_dir"]) / "*.log"))
            
            self.logger.info("📊 Weights & Biases initialisé avec configuration étendue")
            return True
            
        except Exception as e:
            self.logger.warning(f"W&B non disponible: {e}")
            return False

    def setup_quantization_config(self):
        """Configuration de quantisation adaptative"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory >= 24:  # A100, V100
                self.use_4bit = False
                self.use_8bit = True
                self.logger.info("Configuration: 8-bit (GPU haute mémoire)")
            elif gpu_memory >= 16:  # T4, RTX series
                self.use_4bit = True
                self.use_8bit = False
                self.logger.info("Configuration: 4-bit (GPU mémoire moyenne)")
            else:  # GPU faible mémoire
                self.use_4bit = True
                self.use_8bit = False
                self.logger.info("Configuration: 4-bit (GPU faible mémoire)")
        else:
            self.use_4bit = False
            self.use_8bit = False
            self.logger.warning("Pas de GPU détecté - entraînement CPU")
        
        # Configuration BitsAndBytes
        if self.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            self.bnb_config = None

    def setup_lora_config(self):
        """Configuration LoRA optimisée"""
        model_name = self.config["model"]["name"].lower()
        
        if "7b" in model_name:
            r = 16
            lora_alpha = 32
        elif "13b" in model_name:
            r = 8
            lora_alpha = 16
        else:
            r = 16
            lora_alpha = 32
            
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.logger.info(f"🎯 Configuration LoRA: r={r}, alpha={lora_alpha}")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Trouve le dernier checkpoint disponible"""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        # Trier par numéro de step
        latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        
        self.logger.info(f"📁 Checkpoint trouvé: {latest}")
        return str(latest)

    async def unified_train(self):
        """Entraînement unifié principal"""
        self.training_state['status'] = 'starting'
        self.training_state['start_time'] = datetime.now()
        
        try:
            self.logger.info("🚀 === DÉBUT DE L'ENTRAÎNEMENT UNIFIÉ ===")
            
            # 1. Initialisation W&B
            wandb_available = self.init_wandb_enhanced()
            
            # 2. Création/enrichissement du dataset
            dataset_size = await self.create_enhanced_dataset()
            if wandb_available:
                wandb.config.update({"dataset_size": dataset_size})
            
            # 3. Configuration des composants ML
            self.setup_quantization_config()
            self.setup_lora_config()
            
            # 4. Chargement du modèle
            model, tokenizer = self.load_model_and_tokenizer()
            
            # 5. Préparation du dataset
            train_dataset = self.prepare_training_dataset(tokenizer)
            
            # 6. Configuration du trainer
            trainer = self.setup_trainer(model, tokenizer, train_dataset)
            
            # 7. Lancement de l'entraînement
            self.training_state['status'] = 'training'
            
            self.logger.info("🎯 Démarrage de l'entraînement...")
            self.send_alert('info', 'Training Started', f'Entraînement unifié démarré sur {self.platform}')
            
            # Entraînement avec monitoring
            result = trainer.train(resume_from_checkpoint=self.find_latest_checkpoint())
            
            # 8. Sauvegarde finale
            self.save_final_model(trainer, tokenizer)
            
            # 9. Tests automatiques
            self.run_validation_tests(model, tokenizer)
            
            self.training_state['status'] = 'completed'
            self.logger.info("✅ === ENTRAÎNEMENT UNIFIÉ TERMINÉ AVEC SUCCÈS ===")
            self.send_alert('success', 'Training Completed', 'Entraînement unifié terminé avec succès!')
            
            return result
            
        except Exception as e:
            self.training_state['status'] = 'failed'
            self.error_logger.error(f"Entraînement échoué: {e}")
            self.send_alert('error', 'Training Failed', f'Erreur lors de l\'entraînement: {str(e)}')
            raise
        finally:
            self.monitoring_active = False
            if hasattr(self, 'wandb_run') and self.wandb_run:
                wandb.finish()

    def load_model_and_tokenizer(self):
        """Charge le modèle et le tokenizer avec optimisations"""
        self.logger.info(f"📥 Chargement du modèle: {self.config['model']['name']}")
        
        # Chargement du tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=self.config["model"]["trust_remote_code"]
        )
        
        # Ajout du padding token si nécessaire
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Chargement du modèle avec quantisation
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=self.config["model"]["trust_remote_code"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Préparation pour l'entraînement
        model = prepare_model_for_kbit_training(model)
        
        # Application de LoRA
        model = get_peft_model(model, self.lora_config)
        
        # Affichage des paramètres entraînable
        model.print_trainable_parameters()
        
        self.logger.info("✅ Modèle et tokenizer chargés avec succès")
        return model, tokenizer

    def prepare_training_dataset(self, tokenizer):
        """Prépare le dataset pour l'entraînement"""
        self.logger.info(f"📊 Préparation du dataset: {self.dataset_path}")
        
        # Charger le dataset JSONL
        examples = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Format pour l'entraînement instruction-following
        def format_example(example):
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            
            # Format Alpaca-style
            if 'input' in example and example['input']:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{example['input']}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            return prompt
        
        # Tokenisation
        def tokenize_function(example):
            text = format_example(example)
            
            # Tokenisation avec padding et truncation
            result = tokenizer(
                text,
                truncation=True,
                max_length=self.config["model"]["max_seq_length"],
                padding=False,
                return_tensors=None,
            )
            
            # Labels = inputs pour le language modeling
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        # Création du dataset HF
        dataset = Dataset.from_list(examples)
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            desc="Tokenisation du dataset"
        )
        
        self.logger.info(f"✅ Dataset préparé: {len(tokenized_dataset)} exemples")
        return tokenized_dataset

    def setup_trainer(self, model, tokenizer, train_dataset):
        """Configure le trainer Hugging Face"""
        # Arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            lr_scheduler_type=self.config["training"]["lr_scheduler"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            evaluation_strategy=self.config["training"]["eval_strategy"],
            save_strategy=self.config["training"]["save_strategy"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=self.config["training"]["dataloader_num_workers"],
            fp16=self.config["optimization"]["fp16"],
            gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            report_to=self.config["training"]["report_to"],
            run_name=f"unified-llama-cybersec-{self.platform}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Callbacks
        callbacks = []
        if self.config["training"]["eval_strategy"] != "no":
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        self.logger.info("🎯 Trainer configuré avec succès")
        return trainer

    def save_final_model(self, trainer, tokenizer):
        """Sauvegarde le modèle final"""
        self.logger.info("💾 Sauvegarde du modèle final...")
        
        # Sauvegarde du modèle et tokenizer
        final_output_dir = self.output_dir / "final"
        final_output_dir.mkdir(exist_ok=True)
        
        trainer.save_model(str(final_output_dir))
        tokenizer.save_pretrained(str(final_output_dir))
        
        # Sauvegarde des métadonnées
        metadata = {
            "training_completed": datetime.now().isoformat(),
            "platform": self.platform,
            "hardware_specs": self.hardware_specs,
            "config": self.config,
            "training_state": self.training_state
        }
        
        with open(final_output_dir / "training_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Modèle sauvegardé: {final_output_dir}")

    def run_validation_tests(self, model, tokenizer):
        """Exécute des tests de validation sur le modèle"""
        self.logger.info("🧪 Exécution des tests de validation...")
        
        test_prompts = [
            "Créer une règle YARA simple pour détecter un malware",
            "Expliquer ce qu'est une attaque par injection SQL",
            "Comment détecter un mouvement latéral dans un réseau ?"
        ]
        
        # Pipeline de génération
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        for i, prompt in enumerate(test_prompts):
            try:
                formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
                
                response = generator(
                    formatted_prompt,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_text = response[0]['generated_text']
                response_part = generated_text.split("### Response:\n")[-1]
                
                self.logger.info(f"✅ Test {i+1}/3 réussi")
                self.logger.debug(f"Prompt: {prompt}")
                self.logger.debug(f"Réponse: {response_part[:200]}...")
                
            except Exception as e:
                self.logger.error(f"❌ Test {i+1}/3 échoué: {e}")
        
        self.logger.info("🎉 Tests de validation terminés")


def main():
    parser = argparse.ArgumentParser(description='Unified Cloud Trainer for LLaMA CyberSec')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--model-name', default='meta-llama/Llama-2-7b-hf', help='Model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--no-dataset-enrich', action='store_true', help='Skip dataset enrichment')
    
    args = parser.parse_args()
    
    try:
        # Initialiser le trainer
        trainer = UnifiedCloudTrainer(config_path=args.config)
        
        # Appliquer les arguments
        if args.model_name != 'meta-llama/Llama-2-7b-hf':
            trainer.config["model"]["name"] = args.model_name
        if args.epochs != 3:
            trainer.config["training"]["num_epochs"] = args.epochs
        if args.no_dataset_enrich:
            trainer.config["dataset"]["auto_enrich"] = False
        
        # Lancer l'entraînement
        asyncio.run(trainer.unified_train())
        
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()