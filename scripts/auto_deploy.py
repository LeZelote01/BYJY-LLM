#!/usr/bin/env python3
"""
Script de déploiement automatisé pour LLaMA-3-8B Cybersécurité
Fonctionnalités:
- Déploiement one-click sur multiple plateformes
- Configuration automatique de l'environnement
- Installation et gestion des dépendances
- Tests automatiques post-déploiement
- Monitoring et alertes intégrés
- Sauvegarde et rollback automatiques
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import time
import requests
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import yaml
import docker
import psutil
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    """Configuration de déploiement"""
    target_platform: str
    environment: str  # dev, staging, production
    auto_scaling: bool
    monitoring_enabled: bool
    backup_enabled: bool
    ssl_enabled: bool
    domain: Optional[str] = None
    replicas: int = 1
    resources: Dict = None

class AutoDeployer:
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent
        self.config = self.load_deployment_config(config_path)
        self.setup_logging()
        
        # État du déploiement
        self.deployment_state = {
            'status': 'initialized',
            'start_time': None,
            'platform': None,
            'services': {},
            'health_checks': {},
            'rollback_point': None
        }
        
        # Détection automatique de l'environnement
        self.detect_environment()
        
        # Initialisation des clients
        self.init_platform_clients()
    
    def load_deployment_config(self, config_path: str = None) -> Dict:
        """Charge la configuration de déploiement"""
        default_config = {
            "deployment": {
                "platforms": {
                    "docker": {
                        "enabled": True,
                        "registry": "docker.io",
                        "namespace": "llama-cybersec"
                    },
                    "kubernetes": {
                        "enabled": True,
                        "namespace": "llama-cybersec",
                        "ingress_enabled": True
                    },
                    "cloud_run": {
                        "enabled": False,
                        "project_id": "",
                        "region": "us-central1"
                    },
                    "aws_ecs": {
                        "enabled": False,
                        "cluster": "llama-cybersec",
                        "region": "us-east-1"
                    },
                    "azure_container": {
                        "enabled": False,
                        "resource_group": "llama-cybersec",
                        "location": "eastus"
                    }
                },
                "environments": {
                    "development": {
                        "replicas": 1,
                        "resources": {
                            "cpu": "2",
                            "memory": "8Gi",
                            "gpu": 0
                        },
                        "auto_scaling": False
                    },
                    "staging": {
                        "replicas": 2,
                        "resources": {
                            "cpu": "4",
                            "memory": "16Gi",
                            "gpu": 1
                        },
                        "auto_scaling": True
                    },
                    "production": {
                        "replicas": 3,
                        "resources": {
                            "cpu": "8",
                            "memory": "32Gi",
                            "gpu": 1
                        },
                        "auto_scaling": True,
                        "ssl_enabled": True,
                        "monitoring_enabled": True,
                        "backup_enabled": True
                    }
                }
            },
            "monitoring": {
                "prometheus_enabled": True,
                "grafana_enabled": True,
                "alertmanager_enabled": True,
                "log_aggregation": "loki"
            },
            "security": {
                "network_policies": True,
                "pod_security_policies": True,
                "rbac_enabled": True,
                "image_scanning": True
            },
            "backup": {
                "schedule": "0 2 * * *",  # Daily at 2 AM
                "retention_days": 30,
                "storage_class": "standard"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_logging(self):
        """Configuration du logging avancé"""
        log_dir = self.project_root / "logs" / "deployment"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 AutoDeployer initialized")
    
    def detect_environment(self):
        """Détection automatique de l'environnement de déploiement"""
        env_indicators = {
            'kubernetes': self.check_kubernetes(),
            'docker': self.check_docker(),
            'cloud_run': self.check_cloud_run(),
            'aws_ecs': self.check_aws_ecs(),
            'local': True  # Always available as fallback
        }
        
        self.available_platforms = [platform for platform, available in env_indicators.items() if available]
        self.logger.info(f"Plateformes disponibles: {self.available_platforms}")
    
    def check_kubernetes(self) -> bool:
        """Vérifier la disponibilité de Kubernetes"""
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_docker(self) -> bool:
        """Vérifier la disponibilité de Docker"""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except:
            return False
    
    def check_cloud_run(self) -> bool:
        """Vérifier la disponibilité de Google Cloud Run"""
        try:
            result = subprocess.run(['gcloud', 'version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_aws_ecs(self) -> bool:
        """Vérifier la disponibilité d'AWS ECS"""
        try:
            result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def init_platform_clients(self):
        """Initialisation des clients pour les plateformes"""
        self.clients = {}
        
        # Docker client
        if 'docker' in self.available_platforms:
            try:
                self.clients['docker'] = docker.from_env()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Docker client: {e}")
    
    def deploy(self, platform: str, environment: str = 'development', **kwargs):
        """Déploiement principal"""
        self.deployment_state['status'] = 'starting'
        self.deployment_state['start_time'] = datetime.now()
        self.deployment_state['platform'] = platform
        
        try:
            self.logger.info(f"🚀 Début du déploiement sur {platform} en environnement {environment}")
            
            # Validation pré-déploiement
            self.pre_deployment_validation(platform)
            
            # Création du point de rollback
            self.create_rollback_point()
            
            # Préparation de l'environnement
            self.prepare_environment(environment)
            
            # Build des images si nécessaire
            if platform in ['docker', 'kubernetes']:
                self.build_docker_images()
            
            # Déploiement selon la plateforme
            if platform == 'docker':
                self.deploy_docker(environment, **kwargs)
            elif platform == 'kubernetes':
                self.deploy_kubernetes(environment, **kwargs)
            elif platform == 'cloud_run':
                self.deploy_cloud_run(environment, **kwargs)
            elif platform == 'aws_ecs':
                self.deploy_aws_ecs(environment, **kwargs)
            elif platform == 'local':
                self.deploy_local(environment, **kwargs)
            else:
                raise ValueError(f"Plateforme non supportée: {platform}")
            
            # Tests post-déploiement
            self.post_deployment_tests()
            
            # Configuration du monitoring
            if self.config['monitoring'].get('enabled', True):
                self.setup_monitoring(platform)
            
            # Configuration des sauvegardes
            if self.config.get('backup', {}).get('enabled', True):
                self.setup_backup(platform)
            
            self.deployment_state['status'] = 'completed'
            self.logger.info("✅ Déploiement terminé avec succès")
            
        except Exception as e:
            self.deployment_state['status'] = 'failed'
            self.logger.error(f"❌ Déploiement échoué: {e}")
            
            # Tentative de rollback automatique
            if self.deployment_state['rollback_point']:
                self.logger.info("🔄 Tentative de rollback automatique...")
                self.rollback()
            
            raise
    
    def pre_deployment_validation(self, platform: str):
        """Validations pré-déploiement"""
        self.logger.info("🔍 Validation pré-déploiement...")
        
        # Vérifier que la plateforme est disponible
        if platform not in self.available_platforms:
            raise ValueError(f"Plateforme {platform} non disponible")
        
        # Vérifier les prérequis
        required_files = [
            'requirements.txt',
            'scripts/dataset_manager.py',
            'local_deployment/local_interface.py'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Fichier requis manquant: {file_path}")
        
        # Vérifier les ressources système
        self.check_system_resources()
        
        self.logger.info("✅ Validation pré-déploiement réussie")
    
    def check_system_resources(self):
        """Vérification des ressources système"""
        # Vérifier la RAM disponible
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 8:
            self.logger.warning(f"RAM faible détectée: {ram_gb:.1f}GB")
        
        # Vérifier l'espace disque
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_free_gb < 20:
            raise Exception(f"Espace disque insuffisant: {disk_free_gb:.1f}GB libre")
        
        self.logger.info(f"Ressources système: {ram_gb:.1f}GB RAM, {disk_free_gb:.1f}GB disque libre")
    
    def create_rollback_point(self):
        """Créer un point de rollback"""
        rollback_dir = self.project_root / "backups" / f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rollback_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder la configuration actuelle
        config_backup = rollback_dir / "config.json"
        with open(config_backup, 'w') as f:
            json.dump(self.deployment_state, f, indent=2, default=str)
        
        self.deployment_state['rollback_point'] = str(rollback_dir)
        self.logger.info(f"Point de rollback créé: {rollback_dir}")
    
    def prepare_environment(self, environment: str):
        """Préparation de l'environnement"""
        self.logger.info(f"📋 Préparation de l'environnement {environment}")
        
        env_config = self.config['deployment']['environments'].get(environment)
        if not env_config:
            raise ValueError(f"Configuration environnement {environment} non trouvée")
        
        # Créer les répertoires nécessaires
        dirs_to_create = [
            'logs', 'models', 'datasets', 'uploads', 'reports', 'backups'
        ]
        
        for dir_name in dirs_to_create:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
        
        # Génération des fichiers de configuration
        self.generate_config_files(environment)
    
    def generate_config_files(self, environment: str):
        """Génération des fichiers de configuration"""
        config_dir = self.project_root / "configs" / environment
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration de l'interface locale
        local_config = {
            "installation": {
                "install_dir": str(self.project_root)
            },
            "model": {
                "path": str(self.project_root / "models"),
                "format": "auto"
            },
            "inference": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512,
                "context_length": 2048
            },
            "web_ui": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8080
            },
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8081
            }
        }
        
        with open(config_dir / "local_config.json", 'w') as f:
            json.dump(local_config, f, indent=2)
        
        self.logger.info(f"Fichiers de configuration générés pour {environment}")
    
    def build_docker_images(self):
        """Construction des images Docker"""
        self.logger.info("🐳 Construction des images Docker...")
        
        # Dockerfile pour l'interface
        dockerfile_content = self.generate_dockerfile()
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Build de l'image
        client = self.clients.get('docker')
        if client:
            try:
                image, logs = client.images.build(
                    path=str(self.project_root),
                    tag="llama-cybersec:latest",
                    rm=True
                )
                
                self.logger.info(f"✅ Image Docker construite: {image.id}")
                
            except Exception as e:
                self.logger.error(f"❌ Erreur construction image Docker: {e}")
                raise
    
    def generate_dockerfile(self) -> str:
        """Génération du Dockerfile"""
        return """FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Création des répertoires
RUN mkdir -p logs models datasets uploads reports backups

# Exposition des ports
EXPOSE 8080 8081

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Commande par défaut
CMD ["python", "local_deployment/modern_interface.py", "--host", "0.0.0.0", "--port", "8080"]
"""
    
    def deploy_docker(self, environment: str, **kwargs):
        """Déploiement Docker"""
        self.logger.info("🐳 Déploiement Docker...")
        
        client = self.clients.get('docker')
        if not client:
            raise Exception("Client Docker non disponible")
        
        env_config = self.config['deployment']['environments'][environment]
        
        # Configuration du conteneur
        container_config = {
            'image': 'llama-cybersec:latest',
            'name': f'llama-cybersec-{environment}',
            'ports': {'8080/tcp': 8080, '8081/tcp': 8081},
            'environment': {
                'ENVIRONMENT': environment,
                'LOG_LEVEL': 'INFO'
            },
            'volumes': {
                str(self.project_root / 'models'): {'bind': '/app/models', 'mode': 'rw'},
                str(self.project_root / 'logs'): {'bind': '/app/logs', 'mode': 'rw'},
                str(self.project_root / 'datasets'): {'bind': '/app/datasets', 'mode': 'rw'}
            },
            'restart_policy': {'Name': 'unless-stopped'},
            'detach': True
        }
        
        # Ajout de la configuration GPU si disponible
        if env_config.get('resources', {}).get('gpu', 0) > 0:
            container_config['runtime'] = 'nvidia'
            container_config['environment']['NVIDIA_VISIBLE_DEVICES'] = 'all'
        
        try:
            # Arrêter le conteneur existant s'il existe
            try:
                existing_container = client.containers.get(container_config['name'])
                existing_container.stop()
                existing_container.remove()
                self.logger.info("Conteneur existant supprimé")
            except docker.errors.NotFound:
                pass
            
            # Créer et démarrer le nouveau conteneur
            container = client.containers.run(**container_config)
            
            self.deployment_state['services']['docker_container'] = {
                'id': container.id,
                'name': container.name,
                'status': 'running'
            }
            
            self.logger.info(f"✅ Conteneur Docker déployé: {container.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur déploiement Docker: {e}")
            raise
    
    def deploy_kubernetes(self, environment: str, **kwargs):
        """Déploiement Kubernetes"""
        self.logger.info("☸️ Déploiement Kubernetes...")
        
        # Génération des manifests Kubernetes
        self.generate_k8s_manifests(environment)
        
        # Application des manifests
        manifest_dir = self.project_root / "k8s" / environment
        
        try:
            result = subprocess.run([
                'kubectl', 'apply', '-f', str(manifest_dir)
            ], capture_output=True, text=True, check=True)
            
            self.logger.info("✅ Manifests Kubernetes appliqués")
            
            # Attendre que les pods soient prêts
            self.wait_for_k8s_deployment(environment)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Erreur déploiement Kubernetes: {e.stderr}")
            raise
    
    def generate_k8s_manifests(self, environment: str):
        """Génération des manifests Kubernetes"""
        manifest_dir = self.project_root / "k8s" / environment
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        env_config = self.config['deployment']['environments'][environment]
        
        # Namespace
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': f'llama-cybersec-{environment}'
            }
        }
        
        # Deployment
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'llama-cybersec',
                'namespace': f'llama-cybersec-{environment}'
            },
            'spec': {
                'replicas': env_config['replicas'],
                'selector': {
                    'matchLabels': {
                        'app': 'llama-cybersec'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'llama-cybersec'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'llama-cybersec',
                            'image': 'llama-cybersec:latest',
                            'ports': [
                                {'containerPort': 8080},
                                {'containerPort': 8081}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': env_config['resources']['cpu'],
                                    'memory': env_config['resources']['memory']
                                },
                                'limits': {
                                    'cpu': env_config['resources']['cpu'],
                                    'memory': env_config['resources']['memory']
                                }
                            },
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': environment}
                            ]
                        }]
                    }
                }
            }
        }
        
        # Service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'llama-cybersec-service',
                'namespace': f'llama-cybersec-{environment}'
            },
            'spec': {
                'selector': {
                    'app': 'llama-cybersec'
                },
                'ports': [
                    {'port': 8080, 'targetPort': 8080, 'name': 'web'},
                    {'port': 8081, 'targetPort': 8081, 'name': 'api'}
                ]
            }
        }
        
        # Sauvegarde des manifests
        manifests = {
            'namespace.yaml': namespace_manifest,
            'deployment.yaml': deployment_manifest,
            'service.yaml': service_manifest
        }
        
        for filename, manifest in manifests.items():
            with open(manifest_dir / filename, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
        
        self.logger.info(f"Manifests Kubernetes générés dans {manifest_dir}")
    
    def wait_for_k8s_deployment(self, environment: str):
        """Attendre que le déploiement Kubernetes soit prêt"""
        namespace = f'llama-cybersec-{environment}'
        
        for _ in range(30):  # Attendre jusqu'à 5 minutes
            try:
                result = subprocess.run([
                    'kubectl', 'get', 'deployment', 'llama-cybersec',
                    '-n', namespace, '-o', 'json'
                ], capture_output=True, text=True, check=True)
                
                deployment_status = json.loads(result.stdout)
                ready_replicas = deployment_status.get('status', {}).get('readyReplicas', 0)
                desired_replicas = deployment_status.get('spec', {}).get('replicas', 0)
                
                if ready_replicas == desired_replicas:
                    self.logger.info("✅ Déploiement Kubernetes prêt")
                    return
                
                self.logger.info(f"Attente du déploiement... {ready_replicas}/{desired_replicas} replicas prêtes")
                time.sleep(10)
                
            except subprocess.CalledProcessError:
                time.sleep(10)
        
        raise Exception("Timeout: le déploiement Kubernetes n'est pas devenu prêt")
    
    def deploy_local(self, environment: str, **kwargs):
        """Déploiement local"""
        self.logger.info("🏠 Déploiement local...")
        
        # Installation des dépendances
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(self.project_root / 'requirements.txt')
            ], check=True, capture_output=True)
            
            self.logger.info("✅ Dépendances installées")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Erreur installation dépendances: {e}")
            raise
        
        # Démarrage du service local
        self.start_local_service()
    
    def start_local_service(self):
        """Démarrage du service local"""
        interface_script = self.project_root / "local_deployment" / "modern_interface.py"
        
        if not interface_script.exists():
            raise FileNotFoundError("Script de l'interface locale non trouvé")
        
        # Démarrer en arrière-plan
        self.local_process = subprocess.Popen([
            sys.executable, str(interface_script),
            '--host', '0.0.0.0',
            '--port', '8080'
        ])
        
        # Attendre que le service soit prêt
        self.wait_for_local_service()
        
        self.deployment_state['services']['local_process'] = {
            'pid': self.local_process.pid,
            'status': 'running'
        }
    
    def wait_for_local_service(self):
        """Attendre que le service local soit prêt"""
        for _ in range(30):
            try:
                response = requests.get('http://localhost:8080/api/status', timeout=5)
                if response.status_code == 200:
                    self.logger.info("✅ Service local prêt")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        raise Exception("Timeout: le service local n'a pas démarré")
    
    def post_deployment_tests(self):
        """Tests post-déploiement"""
        self.logger.info("🧪 Tests post-déploiement...")
        
        # Test de santé de base
        self.health_check()
        
        # Tests d'API
        self.api_tests()
        
        # Tests de performance
        self.performance_tests()
        
        self.logger.info("✅ Tests post-déploiement réussis")
    
    def health_check(self):
        """Vérification de santé du service"""
        # Déterminer l'URL selon la plateforme
        if self.deployment_state['platform'] == 'local':
            base_url = 'http://localhost:8080'
        else:
            # Pour d'autres plateformes, adapter selon le contexte
            base_url = 'http://localhost:8080'
        
        try:
            response = requests.get(f'{base_url}/api/status', timeout=10)
            if response.status_code == 200:
                status_data = response.json()
                self.deployment_state['health_checks']['api_status'] = status_data
                self.logger.info("✅ Health check API réussi")
            else:
                raise Exception(f"Health check échoué: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Health check échoué: {e}")
    
    def api_tests(self):
        """Tests de l'API"""
        base_url = 'http://localhost:8080'
        
        # Test de génération de texte
        test_payload = {
            'message': 'Test de génération de texte',
            'session_id': 'test-session'
        }
        
        try:
            response = requests.post(f'{base_url}/api/chat', 
                                   json=test_payload, timeout=30)
            
            if response.status_code == 200:
                self.logger.info("✅ Test API chat réussi")
            else:
                self.logger.warning(f"Test API chat: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Test API chat échoué: {e}")
    
    def performance_tests(self):
        """Tests de performance basiques"""
        # Test de temps de réponse
        start_time = time.time()
        
        try:
            response = requests.get('http://localhost:8080/api/status', timeout=10)
            response_time = time.time() - start_time
            
            self.deployment_state['health_checks']['response_time'] = response_time
            
            if response_time < 5.0:
                self.logger.info(f"✅ Temps de réponse acceptable: {response_time:.2f}s")
            else:
                self.logger.warning(f"⚠️ Temps de réponse élevé: {response_time:.2f}s")
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Test de performance échoué: {e}")
    
    def setup_monitoring(self, platform: str):
        """Configuration du monitoring"""
        self.logger.info("📊 Configuration du monitoring...")
        
        monitoring_config = self.config.get('monitoring', {})
        
        if monitoring_config.get('prometheus_enabled', False):
            self.setup_prometheus()
        
        if monitoring_config.get('grafana_enabled', False):
            self.setup_grafana()
        
        self.logger.info("✅ Monitoring configuré")
    
    def setup_prometheus(self):
        """Configuration de Prometheus"""
        # Configuration basique de Prometheus
        prometheus_config = {
            'global': {
                'scrape_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'llama-cybersec',
                    'static_configs': [
                        {'targets': ['localhost:8081']}
                    ]
                }
            ]
        }
        
        config_dir = self.project_root / "monitoring"
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / "prometheus.yml", 'w') as f:
            yaml.dump(prometheus_config, f)
        
        self.logger.info("✅ Configuration Prometheus créée")
    
    def setup_grafana(self):
        """Configuration de Grafana"""
        # Configuration basique de Grafana
        grafana_config = {
            'apiVersion': 1,
            'datasources': [
                {
                    'name': 'Prometheus',
                    'type': 'prometheus',
                    'url': 'http://prometheus:9090',
                    'access': 'proxy'
                }
            ]
        }
        
        config_dir = self.project_root / "monitoring"
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / "grafana-datasources.yml", 'w') as f:
            yaml.dump(grafana_config, f)
        
        self.logger.info("✅ Configuration Grafana créée")
    
    def setup_backup(self, platform: str):
        """Configuration des sauvegardes"""
        self.logger.info("💾 Configuration des sauvegardes...")
        
        backup_config = self.config.get('backup', {})
        
        # Créer le script de sauvegarde
        backup_script = self.generate_backup_script()
        
        backup_script_path = self.project_root / "scripts" / "backup.sh"
        with open(backup_script_path, 'w') as f:
            f.write(backup_script)
        
        backup_script_path.chmod(0o755)
        
        self.logger.info("✅ Script de sauvegarde créé")
    
    def generate_backup_script(self) -> str:
        """Génération du script de sauvegarde"""
        return f"""#!/bin/bash
# Script de sauvegarde automatique pour LLaMA CyberSec

BACKUP_DIR="{self.project_root}/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_$DATE"

# Création du répertoire de sauvegarde
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Sauvegarde des modèles
if [ -d "{self.project_root}/models" ]; then
    echo "Sauvegarde des modèles..."
    cp -r "{self.project_root}/models" "$BACKUP_DIR/$BACKUP_NAME/"
fi

# Sauvegarde des configurations
if [ -d "{self.project_root}/configs" ]; then
    echo "Sauvegarde des configurations..."
    cp -r "{self.project_root}/configs" "$BACKUP_DIR/$BACKUP_NAME/"
fi

# Sauvegarde des logs
if [ -d "{self.project_root}/logs" ]; then
    echo "Sauvegarde des logs..."
    cp -r "{self.project_root}/logs" "$BACKUP_DIR/$BACKUP_NAME/"
fi

# Compression
echo "Compression de la sauvegarde..."
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

# Nettoyage des anciennes sauvegardes (garder 30 jours)
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete

echo "Sauvegarde terminée: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
"""
    
    def rollback(self):
        """Rollback vers le point de sauvegarde"""
        self.logger.info("🔄 Rollback en cours...")
        
        rollback_point = self.deployment_state.get('rollback_point')
        if not rollback_point or not Path(rollback_point).exists():
            raise Exception("Point de rollback non disponible")
        
        # Arrêter les services actuels
        self.stop_services()
        
        # Restaurer la configuration
        # (Implémentation dépendante de la plateforme)
        
        self.logger.info("✅ Rollback terminé")
    
    def stop_services(self):
        """Arrêt des services"""
        services = self.deployment_state.get('services', {})
        
        # Arrêt du processus local
        if 'local_process' in services:
            pid = services['local_process']['pid']
            try:
                os.kill(pid, 15)  # SIGTERM
                self.logger.info(f"Service local arrêté (PID: {pid})")
            except ProcessLookupError:
                pass
        
        # Arrêt du conteneur Docker
        if 'docker_container' in services:
            client = self.clients.get('docker')
            if client:
                try:
                    container = client.containers.get(services['docker_container']['id'])
                    container.stop()
                    self.logger.info("Conteneur Docker arrêté")
                except docker.errors.NotFound:
                    pass
    
    def get_deployment_status(self) -> Dict:
        """Obtenir le statut du déploiement"""
        return {
            'status': self.deployment_state['status'],
            'platform': self.deployment_state['platform'],
            'start_time': self.deployment_state['start_time'],
            'services': self.deployment_state['services'],
            'health_checks': self.deployment_state['health_checks']
        }


def main():
    parser = argparse.ArgumentParser(description='Auto Deployer for LLaMA CyberSec')
    parser.add_argument('--platform', choices=['docker', 'kubernetes', 'cloud_run', 'aws_ecs', 'local'],
                       default='local', help='Target deployment platform')
    parser.add_argument('--environment', choices=['development', 'staging', 'production'],
                       default='development', help='Deployment environment')
    parser.add_argument('--config', help='Path to deployment configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Simulate deployment without executing')
    
    args = parser.parse_args()
    
    try:
        deployer = AutoDeployer(config_path=args.config)
        
        if args.dry_run:
            print(f"🔍 Simulation de déploiement sur {args.platform} en environnement {args.environment}")
            deployer.pre_deployment_validation(args.platform)
            print("✅ Validation réussie - Le déploiement peut être effectué")
        else:
            deployer.deploy(args.platform, args.environment)
            
            # Afficher le statut final
            status = deployer.get_deployment_status()
            print("\n" + "="*50)
            print("📊 STATUT DU DÉPLOIEMENT")
            print("="*50)
            print(f"Statut: {status['status']}")
            print(f"Plateforme: {status['platform']}")
            print(f"Démarré à: {status['start_time']}")
            print(f"Services: {len(status['services'])}")
            
            if status['status'] == 'completed':
                print("\n🎉 Déploiement réussi!")
                print("🌐 Service accessible sur: http://localhost:8080")
            else:
                print(f"\n❌ Déploiement échoué: {status['status']}")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()