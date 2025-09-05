#!/usr/bin/env python3
"""
Script d'entraînement cloud pour LLaMA-3-8B Cybersécurité + Génération de Code
Support: Google Colab, Kaggle, AWS SageMaker, Azure ML
"""

import os
import sys
import json
import torch
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import bitsandbytes as bnb

class CloudTrainingManager:
    def __init__(self, 
                 platform: str = "colab",
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 dataset_path: str = None,
                 output_dir: str = None):
        
        self.platform = platform.lower()
        self.model_name = model_name
        
        # Configuration par plateforme
        self.setup_platform_config()
        
        # Chemins par défaut selon la plateforme
        if dataset_path is None:
            dataset_path = self.get_default_dataset_path()
        if output_dir is None:
            output_dir = self.get_default_output_path()
            
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        
        self.setup_logging()
        self.setup_cloud_storage()
        
        # Configuration d'entraînement optimisée pour le cloud
        self.setup_training_config()
    
    def setup_platform_config(self):
        """Configure selon la plateforme cloud"""
        self.logger = logging.getLogger(__name__)
        
        if self.platform == "colab":
            self.setup_colab()
        elif self.platform == "kaggle":
            self.setup_kaggle()
        elif self.platform == "sagemaker":
            self.setup_sagemaker()
        elif self.platform == "azure":
            self.setup_azure()
        else:
            raise ValueError(f"Plateforme non supportée: {self.platform}")
    
    def setup_colab(self):
        """Configuration Google Colab"""
        self.platform_config = {
            "name": "Google Colab",
            "mount_drive": True,
            "drive_path": "/content/drive/MyDrive",
            "temp_path": "/content",
            "gpu_memory_fraction": 0.9,
            "max_batch_size": 2,
            "use_mixed_precision": True
        }
        
        # Monter Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            self.logger.info("Google Drive monté avec succès")
        except ImportError:
            self.logger.warning("Module Google Colab non disponible")
    
    def setup_kaggle(self):
        """Configuration Kaggle"""
        self.platform_config = {
            "name": "Kaggle",
            "mount_drive": False,
            "temp_path": "/kaggle/working",
            "input_path": "/kaggle/input",
            "gpu_memory_fraction": 0.85,
            "max_batch_size": 1,
            "use_mixed_precision": True
        }
    
    def setup_sagemaker(self):
        """Configuration AWS SageMaker"""
        self.platform_config = {
            "name": "AWS SageMaker",
            "mount_drive": False,
            "temp_path": "/opt/ml/processing",
            "input_path": "/opt/ml/input",
            "output_path": "/opt/ml/output",
            "gpu_memory_fraction": 0.9,
            "max_batch_size": 4,
            "use_mixed_precision": True
        }
    
    def setup_azure(self):
        """Configuration Azure ML"""
        self.platform_config = {
            "name": "Azure ML",
            "mount_drive": False,
            "temp_path": "/tmp",
            "gpu_memory_fraction": 0.9,
            "max_batch_size": 4,
            "use_mixed_precision": True
        }
    
    def setup_logging(self):
        """Configure le logging pour le cloud"""
        log_level = logging.INFO
        
        # Format adapté au cloud
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler pour console (visible dans notebooks)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configuration du logger
        logging.basicConfig(
            level=log_level,
            handlers=[console_handler],
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training sur {self.platform_config['name']}")
    
    def setup_cloud_storage(self):
        """Configure le stockage cloud"""
        if self.platform == "colab" and self.platform_config["mount_drive"]:
            # Utiliser Google Drive pour persistance
            self.persistent_path = Path(self.platform_config["drive_path"]) / "llama_cybersec"
            self.persistent_path.mkdir(exist_ok=True)
        else:
            # Utiliser le stockage temporaire
            self.persistent_path = Path(self.platform_config["temp_path"]) / "llama_cybersec"
            self.persistent_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Stockage configuré: {self.persistent_path}")
    
    def get_default_dataset_path(self):
        """Chemin par défaut du dataset selon la plateforme"""
        if self.platform == "colab":
            return str(self.persistent_path / "dataset" / "enhanced_dataset.jsonl")
        elif self.platform == "kaggle":
            return "/kaggle/input/llama-cybersec-dataset/enhanced_dataset.jsonl"
        else:
            return str(self.persistent_path / "dataset" / "enhanced_dataset.jsonl")
    
    def get_default_output_path(self):
        """Chemin par défaut de sortie selon la plateforme"""
        return str(self.persistent_path / "models" / "lora_weights")
    
    def setup_training_config(self):
        """Configuration d'entraînement optimisée pour le cloud"""
        # Configuration de base adaptée au cloud
        max_batch_size = self.platform_config["max_batch_size"]
        
        self.training_config = {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": max_batch_size * 8,  # Simuler batch plus grand
            "warmup_steps": 100,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "fp16": self.platform_config["use_mixed_precision"],
            "bf16": False,  # Pas toujours supporté sur cloud
            "logging_steps": 10,
            "save_steps": 250,
            "eval_steps": 250,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "output_dir": str(self.output_dir),
            "save_total_limit": 2,  # Limiter l'espace disque
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",  # Désactiver wandb par défaut
            "dataloader_num_workers": 0,  # Éviter les problèmes de multiprocessing
            "group_by_length": True,
            "remove_unused_columns": False,
        }
        
        # Configuration LoRA optimisée
        self.lora_config = LoraConfig(
            r=32,  # Rank plus élevé pour meilleure qualité
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Configuration quantisation
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
    
    def install_dependencies(self):
        """Installe les dépendances nécessaires"""
        self.logger.info("Installation des dépendances...")
        
        packages = [
            "transformers>=4.35.0",
            "datasets>=2.14.0", 
            "peft>=0.6.0",
            "bitsandbytes>=0.41.0",
            "accelerate>=0.24.0",
            "torch>=2.0.0",
            "sentencepiece",
            "protobuf",
        ]
        
        for package in packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ], check=True)
                self.logger.info(f"✓ {package}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"✗ Erreur installation {package}: {e}")
    
    def download_dataset(self):
        """Télécharge ou crée le dataset enrichi"""
        self.logger.info("Préparation du dataset...")
        
        # Créer le répertoire dataset
        dataset_dir = self.dataset_path.parent
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Si le dataset n'existe pas, en créer un enrichi
        if not self.dataset_path.exists():
            self.create_enhanced_dataset()
        
        # Vérifier la taille
        if self.dataset_path.exists():
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for line in f if line.strip())
            self.logger.info(f"Dataset chargé: {lines} exemples")
            return True
        else:
            self.logger.error("Impossible de charger le dataset")
            return False
    
    def create_enhanced_dataset(self):
        """Crée un dataset enrichi pour génération de code et cybersécurité"""
        self.logger.info("Création du dataset enrichi...")
        
        enhanced_examples = []
        
        # Exemples de génération de code complexe
        code_generation_examples = [
            {
                "instruction": "Créer un système complet de détection d'intrusion en Python avec interface web",
                "output": """Voici un système complet de détection d'intrusion avec interface web :

**Structure du projet :**
```
ids_system/
├── app.py                 # Application Flask principale
├── detector/
│   ├── __init__.py
│   ├── network_monitor.py # Monitoring réseau
│   ├── log_analyzer.py    # Analyse de logs
│   └── alert_manager.py   # Gestion des alertes
├── templates/
│   ├── dashboard.html     # Dashboard principal
│   └── alerts.html        # Page des alertes
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
└── requirements.txt

**1. Application principale (app.py) :**
```python
from flask import Flask, render_template, jsonify, request
from detector.network_monitor import NetworkMonitor
from detector.log_analyzer import LogAnalyzer
from detector.alert_manager import AlertManager
import threading
import json

app = Flask(__name__)

# Initialiser les composants
network_monitor = NetworkMonitor()
log_analyzer = LogAnalyzer()
alert_manager = AlertManager()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    stats = {
        'active_connections': network_monitor.get_active_connections_count(),
        'alerts_today': alert_manager.get_alerts_count_today(),
        'suspicious_ips': len(network_monitor.get_suspicious_ips()),
        'cpu_usage': network_monitor.get_cpu_usage()
    }
    return jsonify(stats)

@app.route('/api/alerts')
def get_alerts():
    alerts = alert_manager.get_recent_alerts(limit=50)
    return jsonify(alerts)

@app.route('/api/block_ip', methods=['POST'])
def block_ip():
    ip = request.json.get('ip')
    if network_monitor.block_ip(ip):
        return jsonify({'success': True})
    return jsonify({'success': False})

def start_monitoring():
    # Démarrer le monitoring en arrière-plan
    network_thread = threading.Thread(target=network_monitor.start_monitoring)
    log_thread = threading.Thread(target=log_analyzer.start_analysis)
    
    network_thread.daemon = True
    log_thread.daemon = True
    
    network_thread.start()
    log_thread.start()

if __name__ == '__main__':
    start_monitoring()
    app.run(debug=False, host='0.0.0.0', port=5000)
```

**2. Moniteur réseau (detector/network_monitor.py) :**
```python
import psutil
import socket
import subprocess
import re
from collections import defaultdict, deque
import time
import threading
from datetime import datetime, timedelta

class NetworkMonitor:
    def __init__(self):
        self.suspicious_ips = set()
        self.connection_history = deque(maxlen=1000)
        self.blocked_ips = set()
        self.alert_callback = None
        
    def get_active_connections(self):
        connections = []
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.raddr:
                connections.append({
                    'local': f"{conn.laddr.ip}:{conn.laddr.port}",
                    'remote': f"{conn.raddr.ip}:{conn.raddr.port}",
                    'pid': conn.pid,
                    'status': conn.status
                })
        return connections
    
    def detect_port_scan(self, connections):
        ip_ports = defaultdict(set)
        for conn in connections:
            if conn.get('remote'):
                ip = conn['remote'].split(':')[0]
                port = conn['remote'].split(':')[1]
                ip_ports[ip].add(port)
        
        # Détection: plus de 10 ports différents en 1 minute
        suspicious = []
        for ip, ports in ip_ports.items():
            if len(ports) > 10:
                suspicious.append({
                    'ip': ip,
                    'ports': list(ports),
                    'type': 'port_scan',
                    'severity': 'high'
                })
                self.suspicious_ips.add(ip)
        
        return suspicious
    
    def detect_brute_force(self):
        # Analyser les logs d'authentification
        auth_attempts = defaultdict(int)
        
        try:
            # Lire les logs SSH
            with open('/var/log/auth.log', 'r') as f:
                for line in f.readlines()[-100:]:  # Dernières 100 lignes
                    if 'Failed password' in line:
                        ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', line)
                        if ip_match:
                            ip = ip_match.group(1)
                            auth_attempts[ip] += 1
        except FileNotFoundError:
            pass
        
        # Détection: plus de 5 tentatives échouées
        suspicious = []
        for ip, attempts in auth_attempts.items():
            if attempts > 5:
                suspicious.append({
                    'ip': ip,
                    'attempts': attempts,
                    'type': 'brute_force',
                    'severity': 'critical'
                })
                self.suspicious_ips.add(ip)
        
        return suspicious
    
    def block_ip(self, ip):
        try:
            # Bloquer avec iptables
            subprocess.run([
                'sudo', 'iptables', '-A', 'INPUT', 
                '-s', ip, '-j', 'DROP'
            ], check=True)
            
            self.blocked_ips.add(ip)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def start_monitoring(self):
        while True:
            try:
                connections = self.get_active_connections()
                
                # Détecter les scans de ports
                port_scans = self.detect_port_scan(connections)
                for scan in port_scans:
                    if self.alert_callback:
                        self.alert_callback(scan)
                
                # Détecter le brute force
                brute_forces = self.detect_brute_force()
                for bf in brute_forces:
                    if self.alert_callback:
                        self.alert_callback(bf)
                
                time.sleep(30)  # Vérifier toutes les 30 secondes
                
            except Exception as e:
                print(f"Erreur monitoring: {e}")
                time.sleep(60)
```

**3. Interface web (templates/dashboard.html) :**
```html
<!DOCTYPE html>
<html>
<head>
    <title>IDS Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid">
        <h1 class="mt-3">Système de Détection d'Intrusion</h1>
        
        <div class="row mt-4">
            <div class="col-md-3">
                <div class="card bg-primary text-white">
                    <div class="card-body">
                        <h5>Connexions Actives</h5>
                        <h2 id="active-connections">-</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-danger text-white">
                    <div class="card-body">
                        <h5>Alertes Aujourd'hui</h5>
                        <h2 id="alerts-today">-</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white">
                    <div class="card-body">
                        <h5>IPs Suspectes</h5>
                        <h2 id="suspicious-ips">-</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body">
                        <h5>CPU Usage</h5>
                        <h2 id="cpu-usage">-</h2>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Alertes Récentes</h5>
                    </div>
                    <div class="card-body">
                        <table class="table" id="alerts-table">
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Type</th>
                                    <th>IP Source</th>
                                    <th>Sévérité</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('active-connections').textContent = data.active_connections;
                    document.getElementById('alerts-today').textContent = data.alerts_today;
                    document.getElementById('suspicious-ips').textContent = data.suspicious_ips;
                    document.getElementById('cpu-usage').textContent = data.cpu_usage + '%';
                });
        }
        
        function updateAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.querySelector('#alerts-table tbody');
                    tbody.innerHTML = '';
                    
                    data.forEach(alert => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${alert.timestamp}</td>
                            <td>${alert.type}</td>
                            <td>${alert.ip}</td>
                            <td><span class="badge bg-${alert.severity === 'critical' ? 'danger' : 'warning'}">${alert.severity}</span></td>
                            <td><button class="btn btn-sm btn-danger" onclick="blockIP('${alert.ip}')">Bloquer</button></td>
                        `;
                    });
                });
        }
        
        function blockIP(ip) {
            if (confirm(`Bloquer l'IP ${ip} ?`)) {
                fetch('/api/block_ip', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ip: ip})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`IP ${ip} bloquée avec succès`);
                        updateAlerts();
                    } else {
                        alert('Erreur lors du blocage');
                    }
                });
            }
        }
        
        // Mise à jour automatique
        setInterval(updateStats, 5000);
        setInterval(updateAlerts, 10000);
        
        // Chargement initial
        updateStats();
        updateAlerts();
    </script>
</body>
</html>
```

**Installation et utilisation :**
```bash
# Installation
pip install flask psutil

# Lancement
sudo python app.py  # sudo requis pour iptables

# Accès: http://localhost:5000
```

Ce système fournit une solution complète de détection d'intrusion avec monitoring temps réel, interface web intuitive, et capacités de blocage automatique."""
            },
            
            {
                "instruction": "Développer un framework de test de sécurité automatisé pour applications web avec génération de rapports",
                "output": """Voici un framework complet de test de sécurité automatisé :

**Structure du projet :**
```
web_security_scanner/
├── scanner/
│   ├── __init__.py
│   ├── core/
│   │   ├── scanner_engine.py    # Moteur principal
│   │   ├── vulnerability_db.py  # Base de vulnérabilités
│   │   └── payload_generator.py # Génération de payloads
│   ├── modules/
│   │   ├── sql_injection.py     # Tests injection SQL
│   │   ├── xss_scanner.py       # Tests XSS
│   │   ├── csrf_scanner.py      # Tests CSRF
│   │   └── file_upload.py       # Tests upload
│   └── reporters/
│       ├── html_reporter.py     # Rapport HTML
│       ├── pdf_reporter.py      # Rapport PDF
│       └── json_reporter.py     # Rapport JSON
├── web_interface/
│   ├── app.py                   # Interface web
│   ├── templates/
│   └── static/
└── cli.py                       # Interface ligne de commande
```

**1. Moteur principal (scanner/core/scanner_engine.py) :**
```python
import asyncio
import aiohttp
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging

class SecurityScanner:
    def __init__(self, target_url: str, config: Dict = None):
        self.target_url = target_url.rstrip('/')
        self.base_domain = urlparse(target_url).netloc
        self.session = None
        self.config = config or self.default_config()
        
        # Résultats
        self.vulnerabilities = []
        self.crawled_urls = set()
        self.forms = []
        
        # Modules de test
        self.test_modules = []
        self.load_test_modules()
        
        self.setup_logging()
    
    def default_config(self):
        return {
            'max_depth': 3,
            'max_pages': 100,
            'timeout': 10,
            'concurrent_requests': 5,
            'user_agent': 'WebSecurityScanner/1.0',
            'follow_redirects': True,
            'test_modules': ['sql_injection', 'xss', 'csrf', 'file_upload'],
            'severity_filter': ['low', 'medium', 'high', 'critical']
        }
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_test_modules(self):
        """Charge les modules de test dynamiquement"""
        for module_name in self.config['test_modules']:
            try:
                module = __import__(f'scanner.modules.{module_name}', fromlist=[module_name])
                scanner_class = getattr(module, f'{module_name.title().replace("_", "")}Scanner')
                self.test_modules.append(scanner_class(self))
                self.logger.info(f"Module chargé: {module_name}")
            except Exception as e:
                self.logger.error(f"Erreur chargement module {module_name}: {e}")
    
    async def start_scan(self):
        """Lance le scan complet"""
        self.logger.info(f"Début du scan de {self.target_url}")
        scan_start_time = time.time()
        
        # Créer la session HTTP
        connector = aiohttp.TCPConnector(limit=self.config['concurrent_requests'])
        timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.config['user_agent']}
        ) as session:
            self.session = session
            
            # Phase 1: Crawling
            await self.crawl_website()
            
            # Phase 2: Tests de sécurité
            await self.run_security_tests()
        
        scan_duration = time.time() - scan_start_time
        
        # Génération du rapport
        scan_results = {
            'target': self.target_url,
            'scan_duration': scan_duration,
            'urls_crawled': len(self.crawled_urls),
            'forms_found': len(self.forms),
            'vulnerabilities': self.vulnerabilities,
            'summary': self.generate_summary()
        }
        
        return scan_results
    
    async def crawl_website(self):
        """Crawle le site web pour découvrir les URLs et formulaires"""
        self.logger.info("Phase de crawling...")
        
        urls_to_crawl = [(self.target_url, 0)]
        crawled_count = 0
        
        while urls_to_crawl and crawled_count < self.config['max_pages']:
            # Traitement par batch pour la concurrence
            batch = urls_to_crawl[:self.config['concurrent_requests']]
            urls_to_crawl = urls_to_crawl[self.config['concurrent_requests']:]
            
            tasks = []
            for url, depth in batch:
                if url not in self.crawled_urls and depth <= self.config['max_depth']:
                    tasks.append(self.crawl_single_url(url, depth))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"Erreur crawling: {result}")
                        continue
                    
                    if result:
                        url, new_urls, forms = result
                        self.crawled_urls.add(url)
                        self.forms.extend(forms)
                        urls_to_crawl.extend(new_urls)
                        crawled_count += 1
        
        self.logger.info(f"Crawling terminé: {len(self.crawled_urls)} URLs, {len(self.forms)} formulaires")
    
    async def crawl_single_url(self, url: str, depth: int):
        """Crawle une URL spécifique"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extraire les liens
                new_urls = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    if urlparse(full_url).netloc == self.base_domain:
                        new_urls.append((full_url, depth + 1))
                
                # Extraire les formulaires
                forms = []
                for form in soup.find_all('form'):
                    form_data = self.extract_form_data(form, url)
                    if form_data:
                        forms.append(form_data)
                
                return url, new_urls, forms
                
        except Exception as e:
            self.logger.debug(f"Erreur crawling {url}: {e}")
            return None
    
    def extract_form_data(self, form, base_url):
        """Extrait les données d'un formulaire"""
        action = form.get('action', '')
        method = form.get('method', 'GET').upper()
        action_url = urljoin(base_url, action)
        
        inputs = []
        for input_tag in form.find_all(['input', 'textarea', 'select']):
            input_data = {
                'name': input_tag.get('name', ''),
                'type': input_tag.get('type', 'text'),
                'value': input_tag.get('value', ''),
                'required': input_tag.has_attr('required')
            }
            if input_data['name']:
                inputs.append(input_data)
        
        return {
            'url': action_url,
            'method': method,
            'inputs': inputs,
            'base_url': base_url
        }
    
    async def run_security_tests(self):
        """Exécute les tests de sécurité"""
        self.logger.info("Phase de tests de sécurité...")
        
        for module in self.test_modules:
            self.logger.info(f"Exécution module: {module.__class__.__name__}")
            try:
                module_results = await module.run_tests()
                self.vulnerabilities.extend(module_results)
            except Exception as e:
                self.logger.error(f"Erreur module {module.__class__.__name__}: {e}")
    
    def generate_summary(self):
        """Génère un résumé des vulnérabilités"""
        summary = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        for vuln in self.vulnerabilities:
            severity = vuln.get('severity', 'info')
            summary[severity] = summary.get(severity, 0) + 1
        
        return summary

# Classe de base pour les modules de test
class BaseTestModule:
    def __init__(self, scanner):
        self.scanner = scanner
        self.vulnerabilities = []
    
    async def run_tests(self):
        """Méthode à implémenter par chaque module"""
        raise NotImplementedError
    
    def add_vulnerability(self, vuln_type, severity, url, details):
        """Ajoute une vulnérabilité"""
        vulnerability = {
            'type': vuln_type,
            'severity': severity,
            'url': url,
            'details': details,
            'timestamp': time.time()
        }
        self.vulnerabilities.append(vulnerability)
```

**2. Module injection SQL (scanner/modules/sql_injection.py) :**
```python
import asyncio
from scanner.core.scanner_engine import BaseTestModule
import re

class SqlInjectionScanner(BaseTestModule):
    def __init__(self, scanner):
        super().__init__(scanner)
        
        # Payloads d'injection SQL
        self.payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "'; DROP TABLE users; --",
            "' UNION SELECT 1,2,3 --",
            "1' AND SLEEP(5) --",
            "1' OR 1=1#",
            "admin'--",
            "1' AND (SELECT SUBSTRING(@@version,1,1))='5' --"
        ]
        
        # Patterns d'erreur SQL
        self.error_patterns = [
            r"mysql_fetch_array\(\)",
            r"ORA-[0-9]+",
            r"PostgreSQL.*ERROR",
            r"Warning.*mysql_.*",
            r"valid MySQL result",
            r"MySqlClient\.",
            r"SQLite.*error",
            r"sqlite3\.OperationalError",
            r"Microsoft JET Database",
            r"ODBC Microsoft Access Driver",
            r"Oracle error",
            r"PostgreSQL query failed",
            r"SQL Server.*Driver"
        ]
    
    async def run_tests(self):
        """Exécute les tests d'injection SQL"""
        # Test sur les formulaires
        for form in self.scanner.forms:
            await self.test_form_sql_injection(form)
        
        # Test sur les paramètres URL
        for url in self.scanner.crawled_urls:
            if '?' in url:
                await self.test_url_sql_injection(url)
        
        return self.vulnerabilities
    
    async def test_form_sql_injection(self, form):
        """Teste l'injection SQL sur un formulaire"""
        for payload in self.payloads:
            try:
                # Préparer les données du formulaire
                form_data = {}
                vulnerable_param = None
                
                for input_field in form['inputs']:
                    if input_field['type'] not in ['submit', 'button', 'hidden']:
                        form_data[input_field['name']] = payload
                        vulnerable_param = input_field['name']
                        break
                    else:
                        form_data[input_field['name']] = input_field['value']
                
                if not vulnerable_param:
                    continue
                
                # Envoyer la requête
                if form['method'] == 'POST':
                    async with self.scanner.session.post(form['url'], data=form_data) as response:
                        content = await response.text()
                else:
                    async with self.scanner.session.get(form['url'], params=form_data) as response:
                        content = await response.text()
                
                # Analyser la réponse
                if self.detect_sql_injection(content, payload):
                    self.add_vulnerability(
                        vuln_type="SQL Injection",
                        severity="critical",
                        url=form['url'],
                        details={
                            'parameter': vulnerable_param,
                            'payload': payload,
                            'method': form['method'],
                            'evidence': content[:500]
                        }
                    )
                    break  # Arrêter après la première vulnérabilité trouvée
                    
            except Exception as e:
                continue
    
    async def test_url_sql_injection(self, url):
        """Teste l'injection SQL sur les paramètres URL"""
        from urllib.parse import urlparse, parse_qs
        
        try:
            parsed_url = urlparse(url)
            params = parse_qs(parsed_url.query)
            
            for param_name, param_values in params.items():
                for payload in self.payloads[:3]:  # Limiter les tests sur URL
                    try:
                        # Modifier le paramètre
                        modified_params = params.copy()
                        modified_params[param_name] = [payload]
                        
                        # Construire l'URL de test
                        query_string = '&'.join([f"{k}={v[0]}" for k, v in modified_params.items()])
                        test_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{query_string}"
                        
                        async with self.scanner.session.get(test_url) as response:
                            content = await response.text()
                        
                        if self.detect_sql_injection(content, payload):
                            self.add_vulnerability(
                                vuln_type="SQL Injection",
                                severity="critical",
                                url=test_url,
                                details={
                                    'parameter': param_name,
                                    'payload': payload,
                                    'method': 'GET',
                                    'evidence': content[:500]
                                }
                            )
                            break
                            
                    except Exception:
                        continue
                        
        except Exception as e:
            pass
    
    def detect_sql_injection(self, content, payload):
        """Détecte les signes d'injection SQL"""
        content_lower = content.lower()
        
        # Vérifier les patterns d'erreur
        for pattern in self.error_patterns:
            if re.search(pattern.lower(), content_lower):
                return True
        
        # Vérifier les patterns spécifiques aux payloads
        if "union" in payload.lower():
            if re.search(r'\d+\|\d+\|\d+', content) or re.search(r'1.*2.*3', content):
                return True
        
        if "sleep" in payload.lower():
            # Pour les time-based, on devrait mesurer le temps de réponse
            # Simplifié ici
            return False
        
        return False
```

**3. Générateur de rapports HTML (scanner/reporters/html_reporter.py) :**
```python
from jinja2 import Template
import os
from datetime import datetime

class HTMLReporter:
    def __init__(self):
        self.template = self.get_template()
    
    def generate_report(self, scan_results, output_path):
        """Génère un rapport HTML"""
        
        # Préparer les données pour le template
        report_data = {
            'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_url': scan_results['target'],
            'scan_duration': f"{scan_results['scan_duration']:.2f}s",
            'urls_crawled': scan_results['urls_crawled'],
            'forms_found': scan_results['forms_found'],
            'vulnerabilities': scan_results['vulnerabilities'],
            'summary': scan_results['summary'],
            'total_vulns': len(scan_results['vulnerabilities'])
        }
        
        # Générer le HTML
        html_content = self.template.render(**report_data)
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def get_template(self):
        return Template('''
<!DOCTYPE html>
<html>
<head>
    <title>Rapport de Sécurité Web</title>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .vulnerability-critical { border-left: 5px solid #dc3545; }
        .vulnerability-high { border-left: 5px solid #fd7e14; }
        .vulnerability-medium { border-left: 5px solid #ffc107; }
        .vulnerability-low { border-left: 5px solid #20c997; }
        .vulnerability-info { border-left: 5px solid #6c757d; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <h1 class="mt-4 mb-4">🔒 Rapport de Sécurité Web</h1>
                
                <!-- Informations générales -->
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5>📊 Informations du Scan</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>🎯 Cible:</strong> {{ target_url }}</p>
                                <p><strong>📅 Date:</strong> {{ scan_date }}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>⏱️ Durée:</strong> {{ scan_duration }}</p>
                                <p><strong>🔗 URLs crawlées:</strong> {{ urls_crawled }}</p>
                                <p><strong>📝 Formulaires:</strong> {{ forms_found }}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Résumé des vulnérabilités -->
                <div class="card mb-4">
                    <div class="card-header bg-danger text-white">
                        <h5>⚠️ Résumé des Vulnérabilités</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-8">
                                <div class="row">
                                    <div class="col">
                                        <div class="text-center">
                                            <div class="badge bg-danger fs-4">{{ summary.critical }}</div>
                                            <div class="small">Critiques</div>
                                        </div>
                                    </div>
                                    <div class="col">
                                        <div class="text-center">
                                            <div class="badge bg-warning fs-4">{{ summary.high }}</div>
                                            <div class="small">Élevées</div>
                                        </div>
                                    </div>
                                    <div class="col">
                                        <div class="text-center">
                                            <div class="badge bg-warning fs-4">{{ summary.medium }}</div>
                                            <div class="small">Moyennes</div>
                                        </div>
                                    </div>
                                    <div class="col">
                                        <div class="text-center">
                                            <div class="badge bg-success fs-4">{{ summary.low }}</div>
                                            <div class="small">Faibles</div>
                                        </div>
                                    </div>
                                    <div class="col">
                                        <div class="text-center">
                                            <div class="badge bg-secondary fs-4">{{ summary.info }}</div>
                                            <div class="small">Info</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <canvas id="severityChart" width="200" height="200"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Liste des vulnérabilités -->
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h5>📋 Détail des Vulnérabilités ({{ total_vulns }})</h5>
                    </div>
                    <div class="card-body">
                        {% for vuln in vulnerabilities %}
                        <div class="card mb-3 vulnerability-{{ vuln.severity }}">
                            <div class="card-header">
                                <div class="row align-items-center">
                                    <div class="col">
                                        <h6 class="mb-0">{{ vuln.type }}</h6>
                                    </div>
                                    <div class="col-auto">
                                        <span class="badge bg-{% if vuln.severity == 'critical' %}danger{% elif vuln.severity == 'high' %}warning{% elif vuln.severity == 'medium' %}warning{% elif vuln.severity == 'low' %}success{% else %}secondary{% endif %}">
                                            {{ vuln.severity.upper() }}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body">
                                <p><strong>🔗 URL:</strong> <code>{{ vuln.url }}</code></p>
                                {% if vuln.details.parameter %}
                                <p><strong>📌 Paramètre:</strong> <code>{{ vuln.details.parameter }}</code></p>
                                {% endif %}
                                {% if vuln.details.payload %}
                                <p><strong>💉 Payload:</strong> <code>{{ vuln.details.payload }}</code></p>
                                {% endif %}
                                {% if vuln.details.method %}
                                <p><strong>📤 Méthode:</strong> {{ vuln.details.method }}</p>
                                {% endif %}
                                {% if vuln.details.evidence %}
                                <div class="mt-2">
                                    <strong>🔍 Preuve:</strong>
                                    <pre class="bg-light p-2 small">{{ vuln.details.evidence }}</pre>
                                </div>
                                {% endif %}
                            </div>
                        </div>
                        {% endfor %}
                        
                        {% if total_vulns == 0 %}
                        <div class="alert alert-success text-center">
                            <h4>🎉 Aucune vulnérabilité détectée !</h4>
                            <p>L'application semble sécurisée selon les tests effectués.</p>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Graphique de répartition des sévérités
        const ctx = document.getElementById('severityChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Critiques', 'Élevées', 'Moyennes', 'Faibles', 'Info'],
                datasets: [{
                    data: [{{ summary.critical }}, {{ summary.high }}, {{ summary.medium }}, {{ summary.low }}, {{ summary.info }}],
                    backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#20c997', '#6c757d']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    </script>
</body>
</html>
        ''')
```

**4. Interface ligne de commande (cli.py) :**
```python
#!/usr/bin/env python3
import asyncio
import argparse
import json
from scanner.core.scanner_engine import SecurityScanner
from scanner.reporters.html_reporter import HTMLReporter

async def main():
    parser = argparse.ArgumentParser(description='Scanner de sécurité web automatisé')
    parser.add_argument('url', help='URL cible à scanner')
    parser.add_argument('--output', '-o', default='security_report.html', help='Fichier de rapport')
    parser.add_argument('--format', '-f', choices=['html', 'json'], default='html', help='Format du rapport')
    parser.add_argument('--depth', '-d', type=int, default=3, help='Profondeur de crawling')
    parser.add_argument('--pages', '-p', type=int, default=100, help='Nombre max de pages')
    parser.add_argument('--concurrent', '-c', type=int, default=5, help='Requêtes concurrentes')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'max_depth': args.depth,
        'max_pages': args.pages,
        'concurrent_requests': args.concurrent
    }
    
    # Scanner
    scanner = SecurityScanner(args.url, config)
    results = await scanner.start_scan()
    
    # Génération du rapport
    if args.format == 'html':
        reporter = HTMLReporter()
        output_path = reporter.generate_report(results, args.output)
        print(f"📄 Rapport HTML généré: {output_path}")
    else:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Rapport JSON généré: {args.output}")
    
    # Résumé
    summary = results['summary']
    total_vulns = sum(summary.values())
    
    print(f"\\
🎯 Scan terminé pour {args.url}")
    print(f"🔍 {results['urls_crawled']} URLs analysées")
    print(f"⚠️  {total_vulns} vulnérabilités trouvées:")
    print(f"   🔴 Critiques: {summary['critical']}")
    print(f"   🟠 Élevées: {summary['high']}")
    print(f"   🟡 Moyennes: {summary['medium']}")
    print(f"   🟢 Faibles: {summary['low']}")

if __name__ == '__main__':
    asyncio.run(main())
```

**Installation et utilisation :**
```bash
# Installation
pip install aiohttp beautifulsoup4 jinja2

# Scan basique
python cli.py https://example.com

# Scan avancé avec options
python cli.py https://example.com --depth 5 --pages 200 --concurrent 10 --output detailed_report.html

# Interface web
python web_interface/app.py
```

Ce framework fournit un système complet de test de sécurité avec scanning automatique, détection multi-vulnérabilités, et génération de rapports professionnels."""
            }
        ]
        
        # Exemples d'analyse de code existant
        code_analysis_examples = [
            {
                "instruction": "Analyser ce projet Django et identifier toutes les vulnérabilités de sécurité",
                "output": """Analyse complète du projet Django pour vulnérabilités de sécurité :

**Structure analysée :**
```
django_project/
├── manage.py
├── requirements.txt
├── myproject/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── users/
│   │   ├── models.py
│   │   ├── views.py
│   │   └── forms.py
│   └── blog/
│       ├── models.py
│       ├── views.py
│       └── templates/
└── static/
```

**🚨 VULNÉRABILITÉS CRITIQUES DÉTECTÉES :**

**1. Configuration Settings.py :**
```python
# ❌ PROBLÈME: DEBUG activé en production
DEBUG = True  # CRITIQUE

# ❌ PROBLÈME: SECRET_KEY exposée
SECRET_KEY = 'django-insecure-abc123...'  # PUBLIC

# ❌ PROBLÈME: ALLOWED_HOSTS trop permissif
ALLOWED_HOSTS = ['*']  # CRITIQUE

# ❌ PROBLÈME: Pas de validation CSRF
MIDDLEWARE = [
    # 'django.middleware.csrf.CsrfViewMiddleware',  # MANQUANT
]

# ❌ PROBLÈME: Base de données par défaut
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',  # CRITIQUE pour production
    }
}
```

**🔧 CORRECTIONS RECOMMANDÉES :**
```python
# ✅ CORRIGÉ: Configuration sécurisée
import os
from django.core.exceptions import ImproperlyConfigured

def get_env_variable(var_name):
    try:
        return os.environ[var_name]
    except KeyError:
        error_msg = f"Set the {var_name} environment variable"
        raise ImproperlyConfigured(error_msg)

# ✅ DEBUG basé sur l'environnement
DEBUG = get_env_variable('DEBUG').lower() == 'true'

# ✅ SECRET_KEY depuis variable d'environnement
SECRET_KEY = get_env_variable('SECRET_KEY')

# ✅ ALLOWED_HOSTS restrictif
ALLOWED_HOSTS = get_env_variable('ALLOWED_HOSTS').split(',')

# ✅ Middleware complet
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',  # AJOUTÉ
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# ✅ Configuration sécurité
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000 if not DEBUG else 0
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# ✅ Base de données PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': get_env_variable('DB_NAME'),
        'USER': get_env_variable('DB_USER'),
        'PASSWORD': get_env_variable('DB_PASSWORD'),
        'HOST': get_env_variable('DB_HOST'),
        'PORT': get_env_variable('DB_PORT'),
    }
}
```

**2. Vulnérabilités dans les Vues :**
```python
# ❌ PROBLÈME: Injection SQL brute
def get_user_posts(request):
    user_id = request.GET.get('user_id')
    # VULNÉRABLE À L'INJECTION SQL
    posts = Post.objects.extra(
        where=["user_id = %s"],
        params=[user_id]  # Pas de validation
    )
    return render(request, 'posts.html', {'posts': posts})

# ❌ PROBLÈME: XSS par sortie non échappée
def display_comment(request):
    comment = request.GET.get('comment')
    # VULNÉRABLE AU XSS
    return HttpResponse(f"<div>{comment}</div>")

# ❌ PROBLÈME: Pas de protection CSRF
@csrf_exempt  # DANGEREUX
def delete_post(request):
    if request.method == 'POST':
        post_id = request.POST.get('post_id')
        Post.objects.filter(id=post_id).delete()
        return redirect('home')
```

**🔧 CORRECTIONS :**
```python
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.utils.html import escape
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError

# ✅ CORRIGÉ: Utilisation de l'ORM Django
@login_required
def get_user_posts(request):
    try:
        user_id = int(request.GET.get('user_id', 0))
        if user_id <= 0:
            raise ValidationError("ID utilisateur invalide")
        
        # Utilisation sécurisée de l'ORM
        posts = Post.objects.filter(
            user_id=user_id,
            is_published=True  # Contrôle d'accès
        ).select_related('user')
        
    except (ValueError, ValidationError):
        posts = Post.objects.none()
    
    return render(request, 'posts.html', {'posts': posts})

# ✅ CORRIGÉ: Échappement HTML automatique
def display_comment(request):
    comment = request.GET.get('comment', '')
    # Django échappe automatiquement dans les templates
    return render(request, 'comment.html', {'comment': comment})

# ✅ CORRIGÉ: Protection CSRF et autorisation
@login_required
@require_http_methods(["POST"])
def delete_post(request):
    try:
        post_id = int(request.POST.get('post_id', 0))
        post = get_object_or_404(Post, id=post_id)
        
        # Vérifier que l'utilisateur est propriétaire
        if post.user != request.user and not request.user.is_staff:
            return HttpResponseForbidden("Non autorisé")
        
        post.delete()
        messages.success(request, "Post supprimé avec succès")
        
    except (ValueError, Post.DoesNotExist):
        messages.error(request, "Post introuvable")
    
    return redirect('home')
```

**3. Modèles de Données Vulnérables :**
```python
# ❌ PROBLÈME: Pas de validation des données
class User(models.Model):
    username = models.CharField(max_length=150)
    email = models.EmailField()
    # Pas de validation du mot de passe
    password = models.CharField(max_length=128)

# ❌ PROBLÈME: Permissions trop larges
class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    # Pas de contrôle d'accès
```

**🔧 CORRECTIONS :**
```python
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import re

# ✅ CORRIGÉ: Modèle utilisateur sécurisé
class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    
    def save(self, *args, **kwargs):
        # Validation du mot de passe
        if self.password and not self.password.startswith('pbkdf2_'):
            validate_password(self.password)
        super().save(*args, **kwargs)
    
    def clean(self):
        super().clean()
        # Validation de l'email
        if self.email and User.objects.filter(email=self.email).exclude(pk=self.pk).exists():
            raise ValidationError("Cet email est déjà utilisé")

# ✅ CORRIGÉ: Modèle avec contrôles d'accès
class Post(models.Model):
    title = models.CharField(max_length=200, validators=[validate_post_title])
    content = models.TextField(validators=[validate_post_content])
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    is_published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        permissions = [
            ("can_publish", "Can publish posts"),
            ("can_moderate", "Can moderate posts"),
        ]
    
    def clean(self):
        # Validation du contenu
        if self.content and len(self.content.strip()) < 10:
            raise ValidationError("Le contenu doit faire au moins 10 caractères")

def validate_post_title(value):
    if not re.match(r'^[a-zA-Z0-9\s\-_.,!?]+$', value):
        raise ValidationError("Titre contient des caractères non autorisés")
```

**4. Templates Vulnérables :**
```html
<!-- ❌ PROBLÈME: XSS par |safe non nécessaire -->
<div>{{ user_comment|safe }}</div>

<!-- ❌ PROBLÈME: Pas de protection CSRF -->
<form method="post">
    <input type="hidden" name="action" value="delete">
    <button type="submit">Supprimer</button>
</form>
```

**🔧 CORRECTIONS :**
```html
<!-- ✅ CORRIGÉ: Échappement automatique -->
<div>{{ user_comment }}</div>

<!-- ✅ CORRIGÉ: Protection CSRF -->
<form method="post">
    {% csrf_token %}
    <input type="hidden" name="action" value="delete">
    <button type="submit" onclick="return confirm('Êtes-vous sûr ?')">Supprimer</button>
</form>

<!-- ✅ CORRIGÉ: Validation côté client -->
<form method="post" onsubmit="return validateForm()">
    {% csrf_token %}
    <input type="text" name="title" required maxlength="200">
    <textarea name="content" required minlength="10"></textarea>
    <button type="submit">Publier</button>
</form>
```

**📋 RÉSUMÉ DES CORRECTIONS :**

1. **Configuration** : Variables d'environnement, HTTPS, headers sécurité
2. **Base de données** : PostgreSQL, requêtes préparées
3. **Authentification** : Validation mots de passe, sessions sécurisées  
4. **Autorisation** : Permissions granulaires, contrôles d'accès
5. **Validation** : Côté serveur et client, échappement HTML
6. **Logs** : Monitoring des actions sensibles

**🛡️ TESTS DE SÉCURITÉ RECOMMANDÉS :**
```python
# tests/test_security.py
from django.test import TestCase, Client
from django.contrib.auth import get_user_model

class SecurityTestCase(TestCase):
    def test_csrf_protection(self):
        client = Client(enforce_csrf_checks=True)
        response = client.post('/delete_post/', {'post_id': 1})
        self.assertEqual(response.status_code, 403)
    
    def test_sql_injection_protection(self):
        response = self.client.get('/posts/?user_id=1; DROP TABLE posts;--')
        self.assertEqual(response.status_code, 200)
        # Vérifier que la table existe encore
        self.assertTrue(Post.objects.exists())
    
    def test_xss_protection(self):
        malicious_comment = '<script>alert("XSS")</script>'
        response = self.client.get(f'/comment/?comment={malicious_comment}')
        self.assertNotContains(response, '<script>')
```

**Score de sécurité : 85/100** (après corrections)"""
            }
        ]
        
        # Combiner tous les exemples
        enhanced_examples.extend(code_generation_examples)
        enhanced_examples.extend(code_analysis_examples)
        
        # Sauvegarder le dataset enrichi
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            for example in enhanced_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\
')
        
        self.logger.info(f"Dataset enrichi créé avec {len(enhanced_examples)} exemples")
    
    def setup_wandb(self):
        """Configure Weights & Biases pour le tracking"""
        try:
            import wandb
            
            # Configuration selon la plateforme
            if self.platform == "colab":
                # Dans Colab, on peut utiliser l'API key
                wandb.login()
            
            wandb.init(
                project="llama-cybersec-cloud",
                name=f"training-{self.platform}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "platform": self.platform,
                    "model_name": self.model_name,
                    "dataset_path": str(self.dataset_path),
                    **self.training_config
                }
            )
            
            self.training_config["report_to"] = "wandb"
            self.logger.info("W&B configuré pour tracking")
            
        except ImportError:
            self.logger.warning("W&B non disponible - pas de tracking")
    
    async def start_training(self):
        """Lance l'entraînement complet"""
        self.logger.info("=== DÉBUT DE L'ENTRAÎNEMENT CLOUD ===")
        
        try:
            # 1. Installation des dépendances
            self.install_dependencies()
            
            # 2. Téléchargement/création du dataset
            if not self.download_dataset():
                raise Exception("Impossible de charger le dataset")
            
            # 3. Configuration W&B
            self.setup_wandb()
            
            # 4. Entraînement (utiliser la classe existante mais adaptée)
            from scripts.train_lora import LLaMACyberSecTrainer
            
            trainer = LLaMACyberSecTrainer(
                model_name=self.model_name,
                dataset_path=str(self.dataset_path),
                output_dir=str(self.output_dir),
                use_4bit=True
            )
            
            # Appliquer notre configuration cloud
            trainer.training_config.update(self.training_config)
            trainer.lora_config = self.lora_config
            trainer.bnb_config = self.bnb_config
            
            # Lancer l'entraînement
            trainer.train()
            
            # 5. Sauvegarder sur le stockage persistant
            await self.save_to_persistent_storage()
            
            self.logger.info("=== ENTRAÎNEMENT TERMINÉ ===")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            raise
    
    async def save_to_persistent_storage(self):
        """Sauvegarde les résultats sur le stockage persistant"""
        if self.platform == "colab" and self.platform_config["mount_drive"]:
            # Compresser et sauvegarder sur Google Drive
            import shutil
            
            drive_backup = self.persistent_path / "backup"
            drive_backup.mkdir(exist_ok=True)
            
            # Sauvegarder les poids LoRA
            if self.output_dir.exists():
                backup_path = drive_backup / f"lora_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(self.output_dir, backup_path)
                self.logger.info(f"Poids sauvegardés sur Drive: {backup_path}")
            
            # Créer un archive zip
            shutil.make_archive(
                str(backup_path),
                'zip',
                str(self.output_dir)
            )
            
            self.logger.info("Sauvegarde terminée sur Google Drive")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement cloud LLaMA Cybersécurité')
    parser.add_argument('--platform', choices=['colab', 'kaggle', 'sagemaker', 'azure'], 
                        default='colab', help='Plateforme cloud')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', help='Modèle de base')
    parser.add_argument('--dataset', help='Chemin vers le dataset')
    parser.add_argument('--output', help='Répertoire de sortie')
    parser.add_argument('--epochs', type=int, default=3, help='Nombre d\'époques')
    
    args = parser.parse_args()
    
    # Créer le gestionnaire d'entraînement cloud
    trainer = CloudTrainingManager(
        platform=args.platform,
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output
    )
    
    # Ajuster les époques
    trainer.training_config['num_train_epochs'] = args.epochs
    
    # Lancer l'entraînement
    asyncio.run(trainer.start_training())

if __name__ == "__main__":
    main()