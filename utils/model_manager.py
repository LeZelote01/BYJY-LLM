#!/usr/bin/env python3
"""
Gestionnaire de modèles pour LLaMA-3-8B Cybersécurité
Gestion des modèles entraînés, téléchargement, installation, mise à jour
"""

import os
import sys
import json
import requests
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import subprocess
import argparse

class ModelManager:
    def __init__(self, local_dir: str = None):
        # Configuration des répertoires
        if local_dir is None:
            if os.name == 'nt':  # Windows
                local_dir = os.path.expanduser("~/AppData/Local/LLaMA-CyberSec")
            else:  # Unix/Linux/macOS
                local_dir = os.path.expanduser("~/.llama-cybersec")
        
        self.local_dir = Path(local_dir)
        self.models_dir = self.local_dir / "models"
        self.config_dir = self.local_dir / "config"
        self.cache_dir = self.local_dir / "cache"
        
        # Créer les répertoires
        for directory in [self.models_dir, self.config_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Registre des modèles
        self.registry_file = self.config_dir / "model_registry.json"
        self.load_registry()
        
        # Configuration par défaut
        self.default_sources = {
            "official": "https://huggingface.co/llama-cybersec-official",
            "community": "https://huggingface.co/llama-cybersec-community",
            "github": "https://github.com/CyberSec-LLaMA/models/releases"
        }
    
    def setup_logging(self):
        """Configure le logging"""
        log_file = self.local_dir / "logs" / f"model_manager_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_registry(self):
        """Charge le registre des modèles"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "last_update": None,
                "version": "1.0"
            }
    
    def save_registry(self):
        """Sauvegarde le registre des modèles"""
        self.registry["last_update"] = datetime.now().isoformat()
        
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calcule le hash SHA256 d'un fichier"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def get_available_models(self, source: str = "all") -> Dict[str, Dict]:
        """Récupère la liste des modèles disponibles"""
        available_models = {}
        
        if source == "all" or source == "local":
            # Modèles installés localement
            local_models = self.get_local_models()
            available_models.update(local_models)
        
        if source == "all" or source == "remote":
            # Modèles disponibles en ligne
            remote_models = self.get_remote_models()
            available_models.update(remote_models)
        
        return available_models
    
    def get_local_models(self) -> Dict[str, Dict]:
        """Récupère les modèles installés localement"""
        local_models = {}
        
        for model_id, model_info in self.registry["models"].items():
            if model_info.get("installed", False):
                model_path = Path(model_info["local_path"])
                
                if model_path.exists():
                    # Vérifier l'intégrité
                    if self.verify_model_integrity(model_path, model_info.get("hash")):
                        model_info["status"] = "installed"
                        model_info["size"] = model_path.stat().st_size
                    else:
                        model_info["status"] = "corrupted"
                else:
                    model_info["status"] = "missing"
                
                local_models[model_id] = model_info
        
        return local_models
    
    def get_remote_models(self) -> Dict[str, Dict]:
        """Récupère la liste des modèles disponibles en ligne"""
        remote_models = {}
        
        # Modèles depuis Hugging Face
        hf_models = self.fetch_huggingface_models()
        remote_models.update(hf_models)
        
        # Modèles depuis GitHub Releases
        github_models = self.fetch_github_models()
        remote_models.update(github_models)
        
        return remote_models
    
    def fetch_huggingface_models(self) -> Dict[str, Dict]:
        """Récupère les modèles depuis Hugging Face"""
        models = {}
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Chercher les modèles avec le tag "llama-cybersec"
            repos = api.list_models(
                filter="llama",
                search="cybersec",
                limit=20
            )
            
            for repo in repos:
                model_id = f"hf_{repo.modelId.replace('/', '_')}"
                
                models[model_id] = {
                    "name": repo.modelId,
                    "source": "huggingface",
                    "url": f"https://huggingface.co/{repo.modelId}",
                    "tags": repo.tags,
                    "downloads": getattr(repo, 'downloads', 0),
                    "last_modified": getattr(repo, 'lastModified', None),
                    "description": getattr(repo, 'cardData', {}).get('description', ''),
                    "installed": False
                }
        
        except ImportError:
            self.logger.warning("huggingface_hub non installé")
        except Exception as e:
            self.logger.error(f"Erreur récupération modèles HF: {e}")
        
        return models
    
    def fetch_github_models(self) -> Dict[str, Dict]:
        """Récupère les modèles depuis GitHub Releases"""
        models = {}
        
        try:
            # API GitHub pour récupérer les releases
            api_url = "https://api.github.com/repos/CyberSec-LLaMA/models/releases"
            
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            releases = response.json()
            
            for release in releases[:10]:  # Limiter à 10 releases
                for asset in release.get("assets", []):
                    if asset["name"].endswith((".gguf", ".bin")):
                        model_id = f"gh_{asset['name'].replace('.', '_')}"
                        
                        models[model_id] = {
                            "name": asset["name"],
                            "source": "github",
                            "url": asset["browser_download_url"],
                            "size": asset["size"],
                            "download_count": asset["download_count"],
                            "release_tag": release["tag_name"],
                            "published_at": release["published_at"],
                            "description": release.get("body", ""),
                            "installed": False
                        }
        
        except Exception as e:
            self.logger.error(f"Erreur récupération modèles GitHub: {e}")
        
        return models
    
    def install_model(self, 
                     model_id: str, 
                     source_url: str = None,
                     force_reinstall: bool = False) -> bool:
        """Installe un modèle"""
        
        # Vérifier si le modèle est déjà installé
        if model_id in self.registry["models"] and not force_reinstall:
            model_info = self.registry["models"][model_id]
            if model_info.get("installed", False):
                model_path = Path(model_info["local_path"])
                if model_path.exists() and self.verify_model_integrity(model_path, model_info.get("hash")):
                    self.logger.info(f"Modèle {model_id} déjà installé")
                    return True
        
        # Récupérer les informations du modèle
        if not source_url:
            available_models = self.get_remote_models()
            if model_id not in available_models:
                self.logger.error(f"Modèle {model_id} non trouvé")
                return False
            
            model_info = available_models[model_id]
            source_url = model_info["url"]
        else:
            model_info = {"name": model_id, "url": source_url}
        
        self.logger.info(f"Installation du modèle {model_id}...")
        
        try:
            # Déterminer le nom de fichier local
            filename = self.get_model_filename(model_id, source_url)
            local_path = self.models_dir / "gguf" / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Télécharger le modèle
            if not self.download_model(source_url, local_path):
                return False
            
            # Calculer le hash
            file_hash = self.calculate_file_hash(local_path)
            
            # Mettre à jour le registre
            self.registry["models"][model_id] = {
                **model_info,
                "local_path": str(local_path),
                "installed": True,
                "installed_at": datetime.now().isoformat(),
                "hash": file_hash,
                "size": local_path.stat().st_size
            }
            
            self.save_registry()
            self.logger.info(f"Modèle {model_id} installé avec succès")
            
            # Post-installation
            self.post_install_setup(model_id, local_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur installation modèle {model_id}: {e}")
            return False
    
    def get_model_filename(self, model_id: str, source_url: str) -> str:
        """Détermine le nom de fichier pour un modèle"""
        # Extraire le nom de fichier de l'URL
        filename = source_url.split("/")[-1]
        
        # S'assurer qu'il a une extension valide
        if not any(filename.endswith(ext) for ext in [".gguf", ".bin", ".safetensors"]):
            filename += ".gguf"
        
        # Préfixer avec l'ID du modèle pour éviter les conflits
        safe_model_id = model_id.replace("/", "_").replace(" ", "_")
        return f"{safe_model_id}_{filename}"
    
    def download_model(self, url: str, local_path: Path) -> bool:
        """Télécharge un modèle avec barre de progression"""
        try:
            self.logger.info(f"Téléchargement depuis {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    chunk_size = 8192
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Afficher la progression
                            progress = (downloaded / total_size) * 100
                            print(f"\rTéléchargement: {progress:.1f}% ({downloaded}/{total_size} bytes)", 
                                  end='', flush=True)
            
            print()  # Nouvelle ligne
            self.logger.info(f"Téléchargement terminé: {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur téléchargement: {e}")
            if local_path.exists():
                local_path.unlink()  # Supprimer le fichier partiel
            return False
    
    def verify_model_integrity(self, model_path: Path, expected_hash: str = None) -> bool:
        """Vérifie l'intégrité d'un modèle"""
        if not model_path.exists():
            return False
        
        # Vérification basique de la taille
        if model_path.stat().st_size < 1024 * 1024:  # < 1MB
            self.logger.warning(f"Taille suspecte pour {model_path}")
            return False
        
        # Vérification du hash si fourni
        if expected_hash:
            actual_hash = self.calculate_file_hash(model_path)
            if actual_hash != expected_hash:
                self.logger.error(f"Hash incorrect pour {model_path}")
                return False
        
        # Vérification du format GGUF si applicable
        if model_path.suffix == '.gguf':
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'GGUF'):
                        self.logger.error(f"Format GGUF invalide: {model_path}")
                        return False
            except Exception:
                return False
        
        return True
    
    def post_install_setup(self, model_id: str, model_path: Path):
        """Configuration post-installation"""
        try:
            # Créer un fichier de configuration pour le modèle
            config = {
                "model_id": model_id,
                "model_path": str(model_path),
                "model_type": self.detect_model_type(model_path),
                "recommended_settings": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 512,
                    "context_length": 2048
                },
                "installed_at": datetime.now().isoformat()
            }
            
            config_file = self.config_dir / f"{model_id}_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Créer un script de lancement si c'est le premier modèle
            if len([m for m in self.registry["models"].values() if m.get("installed")]) == 1:
                self.create_launch_script(model_id)
            
            self.logger.info(f"Configuration post-installation terminée pour {model_id}")
            
        except Exception as e:
            self.logger.warning(f"Erreur configuration post-installation: {e}")
    
    def detect_model_type(self, model_path: Path) -> str:
        """Détecte le type de modèle"""
        suffix = model_path.suffix.lower()
        
        if suffix in ['.gguf']:
            return "gguf"
        elif suffix in ['.bin', '.safetensors']:
            return "transformers"
        else:
            return "unknown"
    
    def create_launch_script(self, model_id: str):
        """Crée un script de lancement pour le modèle"""
        script_content = f"""#!/usr/bin/env python3
# Script de lancement automatique pour {model_id}

import sys
from pathlib import Path

# Ajouter le répertoire local_deployment au path
sys.path.insert(0, str(Path(__file__).parent / "local_deployment"))

from local_interface import LocalLLaMAInterface

if __name__ == "__main__":
    try:
        interface = LocalLLaMAInterface()
        print(f"🚀 Démarrage de LLaMA CyberSec avec le modèle {model_id}")
        interface.run()
    except Exception as e:
        print(f"❌ Erreur: {{e}}")
        input("Appuyez sur Entrée pour fermer...")
"""
        
        if os.name == 'nt':  # Windows
            script_path = self.local_dir / "launch.bat"
            bat_content = f"""@echo off
cd /d "{self.local_dir}"
python launch_model.py
pause
"""
            with open(script_path, 'w') as f:
                f.write(bat_content)
        else:  # Unix/Linux/macOS
            script_path = self.local_dir / "launch.sh"
            with open(script_path, 'w') as f:
                f.write(f"""#!/bin/bash
cd "{self.local_dir}"
python3 launch_model.py
""")
            script_path.chmod(0o755)
        
        # Script Python principal
        py_script_path = self.local_dir / "launch_model.py"
        with open(py_script_path, 'w') as f:
            f.write(script_content)
        
        self.logger.info(f"Script de lancement créé: {script_path}")
    
    def uninstall_model(self, model_id: str) -> bool:
        """Désinstalle un modèle"""
        if model_id not in self.registry["models"]:
            self.logger.error(f"Modèle {model_id} non trouvé dans le registre")
            return False
        
        model_info = self.registry["models"][model_id]
        
        try:
            # Supprimer le fichier du modèle
            if "local_path" in model_info:
                model_path = Path(model_info["local_path"])
                if model_path.exists():
                    model_path.unlink()
                    self.logger.info(f"Fichier modèle supprimé: {model_path}")
            
            # Supprimer la configuration
            config_file = self.config_dir / f"{model_id}_config.json"
            if config_file.exists():
                config_file.unlink()
            
            # Mettre à jour le registre
            self.registry["models"][model_id]["installed"] = False
            self.registry["models"][model_id]["uninstalled_at"] = datetime.now().isoformat()
            
            self.save_registry()
            self.logger.info(f"Modèle {model_id} désinstallé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur désinstallation modèle {model_id}: {e}")
            return False
    
    def update_model(self, model_id: str) -> bool:
        """Met à jour un modèle installé"""
        if model_id not in self.registry["models"]:
            self.logger.error(f"Modèle {model_id} non installé")
            return False
        
        self.logger.info(f"Mise à jour du modèle {model_id}...")
        
        # Sauvegarder l'ancienne version
        model_info = self.registry["models"][model_id]
        old_path = Path(model_info["local_path"])
        backup_path = old_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            if old_path.exists():
                shutil.copy2(old_path, backup_path)
                self.logger.info(f"Sauvegarde créée: {backup_path}")
            
            # Réinstaller la nouvelle version
            if self.install_model(model_id, force_reinstall=True):
                # Supprimer la sauvegarde si tout s'est bien passé
                if backup_path.exists():
                    backup_path.unlink()
                
                self.logger.info(f"Modèle {model_id} mis à jour avec succès")
                return True
            else:
                # Restaurer la sauvegarde en cas d'échec
                if backup_path.exists() and not old_path.exists():
                    shutil.move(backup_path, old_path)
                    self.logger.info("Ancienne version restaurée")
                
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur mise à jour modèle {model_id}: {e}")
            
            # Restaurer la sauvegarde
            if backup_path.exists() and not old_path.exists():
                try:
                    shutil.move(backup_path, old_path)
                    self.logger.info("Ancienne version restaurée")
                except Exception:
                    pass
            
            return False
    
    def list_models(self, show_remote: bool = False) -> List[Dict]:
        """Liste tous les modèles disponibles"""
        models = []
        
        # Modèles locaux
        local_models = self.get_local_models()
        for model_id, model_info in local_models.items():
            models.append({
                "id": model_id,
                "name": model_info.get("name", model_id),
                "status": model_info.get("status", "unknown"),
                "size": model_info.get("size", 0),
                "installed_at": model_info.get("installed_at"),
                "source": "local"
            })
        
        # Modèles distants si demandé
        if show_remote:
            remote_models = self.get_remote_models()
            for model_id, model_info in remote_models.items():
                if model_id not in local_models:  # Éviter les doublons
                    models.append({
                        "id": model_id,
                        "name": model_info.get("name", model_id),
                        "status": "available",
                        "size": model_info.get("size", 0),
                        "description": model_info.get("description", ""),
                        "source": model_info.get("source", "remote")
                    })
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Récupère les informations détaillées d'un modèle"""
        if model_id in self.registry["models"]:
            return self.registry["models"][model_id]
        
        # Chercher dans les modèles distants
        remote_models = self.get_remote_models()
        if model_id in remote_models:
            return remote_models[model_id]
        
        return None
    
    def cleanup_cache(self) -> bool:
        """Nettoie le cache et les fichiers temporaires"""
        try:
            # Supprimer les fichiers de sauvegarde anciens (> 7 jours)
            for backup_file in self.models_dir.rglob("*.backup_*"):
                if backup_file.stat().st_mtime < (datetime.now().timestamp() - 7 * 24 * 3600):
                    backup_file.unlink()
                    self.logger.info(f"Sauvegarde supprimée: {backup_file}")
            
            # Nettoyer le cache
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.iterdir():
                    if cache_file.stat().st_mtime < (datetime.now().timestamp() - 24 * 3600):
                        if cache_file.is_file():
                            cache_file.unlink()
                        elif cache_file.is_dir():
                            shutil.rmtree(cache_file)
            
            self.logger.info("Nettoyage du cache terminé")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur nettoyage cache: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Gestionnaire de modèles LLaMA CyberSec')
    parser.add_argument('--action', choices=['list', 'install', 'uninstall', 'update', 'info', 'cleanup'],
                       required=True, help='Action à effectuer')
    parser.add_argument('--model-id', help='ID du modèle')
    parser.add_argument('--source-url', help='URL source du modèle')
    parser.add_argument('--show-remote', action='store_true', help='Afficher les modèles distants')
    parser.add_argument('--force', action='store_true', help='Forcer l\'action')
    parser.add_argument('--local-dir', help='Répertoire local')
    
    args = parser.parse_args()
    
    try:
        manager = ModelManager(local_dir=args.local_dir)
        
        if args.action == 'list':
            models = manager.list_models(show_remote=args.show_remote)
            
            print(f"\n{'ID':<30} {'Nom':<40} {'Statut':<15} {'Taille':<15}")
            print("-" * 100)
            
            for model in models:
                size_str = f"{model['size'] / (1024**3):.1f} GB" if model['size'] > 0 else "N/A"
                print(f"{model['id']:<30} {model['name']:<40} {model['status']:<15} {size_str:<15}")
        
        elif args.action == 'install':
            if not args.model_id:
                print("❌ --model-id requis pour l'installation")
                sys.exit(1)
            
            success = manager.install_model(args.model_id, args.source_url, args.force)
            print(f"{'✅' if success else '❌'} Installation {'réussie' if success else 'échouée'}")
        
        elif args.action == 'uninstall':
            if not args.model_id:
                print("❌ --model-id requis pour la désinstallation")
                sys.exit(1)
            
            success = manager.uninstall_model(args.model_id)
            print(f"{'✅' if success else '❌'} Désinstallation {'réussie' if success else 'échouée'}")
        
        elif args.action == 'update':
            if not args.model_id:
                print("❌ --model-id requis pour la mise à jour")
                sys.exit(1)
            
            success = manager.update_model(args.model_id)
            print(f"{'✅' if success else '❌'} Mise à jour {'réussie' if success else 'échouée'}")
        
        elif args.action == 'info':
            if not args.model_id:
                print("❌ --model-id requis pour les informations")
                sys.exit(1)
            
            info = manager.get_model_info(args.model_id)
            if info:
                print(f"\n=== Informations du modèle {args.model_id} ===")
                for key, value in info.items():
                    print(f"{key}: {value}")
            else:
                print(f"❌ Modèle {args.model_id} non trouvé")
        
        elif args.action == 'cleanup':
            success = manager.cleanup_cache()
            print(f"{'✅' if success else '❌'} Nettoyage {'terminé' if success else 'échoué'}")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()