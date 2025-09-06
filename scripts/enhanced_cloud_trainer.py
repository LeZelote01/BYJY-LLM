#!/usr/bin/env python3
"""
Script d'entraînement cloud ultra-optimisé pour LLaMA-3-8B Cybersécurité
Nouvelles fonctionnalités:
- Auto-détection de ressources et optimisation adaptative
- Reprise intelligente d'entraînement
- Monitoring en temps réel avec alertes
- Déploiement automatique multi-cloud
- Validation et tests automatiques
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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import requests
import wandb
import threading
from queue import Queue
import smtplib
from email.mime.text import MIMEText

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
import bitsandbytes as bnb

class EnhancedCloudTrainer:
    def __init__(self, config_path: str = None):
        # Chargement de la configuration
        self.config = self.load_configuration(config_path)
        
        # Détection automatique de l'environnement
        self.platform = self.detect_platform()
        self.hardware_specs = self.analyze_hardware()
        
        # Optimisation automatique selon l'environnement
        self.auto_optimize_config()
        
        # Initialisation des composants
        self.setup_logging()
        self.setup_monitoring()
        self.setup_alerts()
        
        # État de l'entraînement
        self.training_state = {
            'status': 'initialized',
            'start_time': None,
            'last_checkpoint': None,
            'metrics_history': [],
            'alerts_sent': []
        }
        
        # Gestionnaire de reproductibilité
        self.setup_reproducibility()
        
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
                "report_to": ["wandb", "tensorboard"]
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
                "wandb_project": "llama-cybersec-enhanced",
                "log_predictions": True,
                "track_gpu_memory": True,
                "alert_on_error": True,
                "alert_on_performance_drop": True
            },
            "paths": {
                "dataset": "/app/datasets/enriched_dataset.jsonl",
                "output_dir": "/app/models/enhanced_lora",
                "checkpoint_dir": "/app/checkpoints",
                "logs_dir": "/app/logs/enhanced_training"
            },
            "cloud": {
                "auto_upload": True,
                "backup_frequency": "hourly",
                "storage_providers": ["s3", "gcs", "azure"],
                "compression": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
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
        
        # Local/Generic
        return "local"
    
    def analyze_hardware(self) -> Dict:
        """Analyse détaillée du hardware disponible"""
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
        """Optimisation automatique de la configuration selon le hardware"""
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
            self.config["model"]["max_seq_length"] = 1024  # Réduire la longueur
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
            self.config["training"]["save_steps"] = 250  # Sauvegardes plus fréquentes
            self.config["cloud"]["backup_frequency"] = "every_checkpoint"
        elif self.platform in ["aws", "gcp", "azure"]:
            self.config["training"]["save_steps"] = 500
            self.config["cloud"]["backup_frequency"] = "hourly"
        
        self.logger.info(f"Configuration auto-optimisée pour {self.platform} avec {gpu_memory_total}GB GPU")
    
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
                logging.FileHandler(log_dir / f'enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Logger spécialisés
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
        
        self.logger.info(f"Logging configuré - Plateforme: {self.platform}")
        self.logger.info(f"Hardware specs: {self.hardware_specs}")
    
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
                
                # Log des métriques critiques
                if cpu_percent > 90:
                    self.logger.warning(f"CPU usage élevé: {cpu_percent}%")
                if ram_percent > 90:
                    self.logger.warning(f"RAM usage élevé: {ram_percent}%")
                
                for gpu_stat in gpu_stats:
                    if gpu_stat['utilization_percent'] > 95:
                        self.logger.warning(f"GPU {gpu_stat['device']} saturé: {gpu_stat['utilization_percent']:.1f}%")
                
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
        
        # Email
        if self.alert_config['email_enabled'] and self.alert_config['email_sender']:
            self._send_email_alert(alert_data)
        
        # Discord
        if self.alert_config['discord_webhook']:
            self._send_discord_alert(alert_data)
        
        # Slack
        if self.alert_config['slack_webhook']:
            self._send_slack_alert(alert_data)
        
        # Webhook générique
        if self.alert_config['webhook_url']:
            self._send_webhook_alert(alert_data)
    
    def _send_email_alert(self, alert_data: Dict):
        """Envoie une alerte par email"""
        try:
            msg = MIMEText(f"""
Alerte LLaMA Training - {alert_data['level'].upper()}

Titre: {alert_data['title']}
Message: {alert_data['message']}
Timestamp: {alert_data['timestamp']}
Plateforme: {alert_data['platform']}

Spécifications Hardware:
- GPU: {alert_data['hardware']['gpu_count']} x {alert_data['hardware']['gpu_memory_total']}GB
- RAM: {alert_data['hardware']['ram_available_gb']}GB disponible
- CPU: {alert_data['hardware']['cpu_count']} cores
            """)
            
            msg['Subject'] = f"[LLaMA Training] {alert_data['level'].upper()}: {alert_data['title']}"
            msg['From'] = self.alert_config['email_sender']
            msg['To'] = ', '.join(self.alert_config['email_recipients'])
            
            with smtplib.SMTP(self.alert_config['email_smtp_server'], self.alert_config['email_smtp_port']) as server:
                server.starttls()
                server.login(self.alert_config['email_sender'], self.alert_config['email_password'])
                server.send_message(msg)
                
            self.logger.info("Alerte email envoyée")
            
        except Exception as e:
            self.error_logger.error(f"Erreur envoi email: {e}")
    
    def _send_discord_alert(self, alert_data: Dict):
        """Envoie une alerte Discord"""
        try:
            embed = {
                "title": f"🚨 {alert_data['title']}",
                "description": alert_data['message'],
                "color": 16711680 if alert_data['level'] == 'error' else 16776960,  # Rouge ou jaune
                "fields": [
                    {"name": "Plateforme", "value": alert_data['platform'], "inline": True},
                    {"name": "GPU", "value": f"{alert_data['hardware']['gpu_count']} x {alert_data['hardware']['gpu_memory_total']}GB", "inline": True},
                    {"name": "RAM", "value": f"{alert_data['hardware']['ram_available_gb']}GB", "inline": True}
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
        
        # Déterminisme CUDA (peut ralentir)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.logger.info(f"Reproductibilité configurée avec seed: {seed}")
    
    def enhanced_train(self):
        """Entraînement amélioré avec monitoring et alertes"""
        self.training_state['status'] = 'starting'
        self.training_state['start_time'] = datetime.now()
        
        try:
            self.logger.info("=== DÉBUT DE L'ENTRAÎNEMENT AMÉLIORÉ ===")
            
            # Initialisation Weights & Biases
            self.init_wandb_enhanced()
            
            # Préparation des données
            dataset = self.load_and_validate_dataset()
            
            # Chargement du modèle avec optimisations
            model, tokenizer = self.load_optimized_model()
            
            # Préparation du dataset
            tokenized_dataset = self.prepare_dataset(dataset, tokenizer)
            
            # Configuration du trainer avancé
            trainer = self.setup_enhanced_trainer(model, tokenizer, tokenized_dataset)
            
            # Validation pré-entraînement
            self.pre_training_validation(trainer)
            
            # Lancement de l'entraînement
            self.training_state['status'] = 'training'
            
            self.logger.info("🚀 Démarrage de l'entraînement...")
            self.send_alert('info', 'Training Started', f'Entraînement démarré sur {self.platform}')
            
            # Entraînement avec monitoring
            result = trainer.train(resume_from_checkpoint=self.find_latest_checkpoint())
            
            # Post-traitement
            self.post_training_analysis(trainer, result)
            
            # Sauvegarde finale
            self.save_final_model(trainer, tokenizer)
            
            # Tests automatiques
            self.automated_testing(model, tokenizer)
            
            self.training_state['status'] = 'completed'
            self.logger.info("=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
            self.send_alert('success', 'Training Completed', 'Entraînement terminé avec succès!')
            
            return result
            
        except Exception as e:
            self.training_state['status'] = 'failed'
            self.error_logger.error(f"Entraînement échoué: {e}")
            self.send_alert('error', 'Training Failed', f'Erreur lors de l\'entraînement: {str(e)}')
            raise
        finally:
            self.monitoring_active = False
            if hasattr(self, 'wandb_run'):
                wandb.finish()
    
    def init_wandb_enhanced(self):
        """Initialisation avancée de Weights & Biases"""
        config_wandb = {
            **self.config,
            'hardware_specs': self.hardware_specs,
            'platform': self.platform,
            'training_start': datetime.now().isoformat()
        }
        
        self.wandb_run = wandb.init(
            project=self.config["monitoring"]["wandb_project"],
            config=config_wandb,
            name=f"enhanced-llama-cybersec-{self.platform}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=[
                self.platform, 
                "cybersecurity", 
                "llama", 
                "lora", 
                "enhanced",
                f"gpu-{self.hardware_specs['gpu_count']}x{self.hardware_specs['gpu_memory_total']}gb"
            ],
            notes=f"Enhanced training on {self.platform} with {self.hardware_specs['gpu_count']} GPUs"
        )
        
        # Log des artefacts de configuration
        wandb.save(str(Path(self.config["paths"]["logs_dir"]) / "*.log"))
        
        self.logger.info("Weights & Biases initialisé avec configuration étendue")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Trouve le dernier checkpoint disponible"""
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        # Trier par numéro de step
        latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        
        self.logger.info(f"Checkpoint trouvé: {latest}")
        return str(latest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Cloud Trainer for LLaMA CyberSec')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    trainer = EnhancedCloudTrainer(config_path=args.config)
    trainer.enhanced_train()