#!/usr/bin/env python3
"""
Installateur local pour LLaMA-3-8B Cybersécurité
Télécharge et configure le modèle entraîné en cloud pour utilisation locale
Support: CPU optimisé, GPU léger, interface web intégrée
"""

import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse
import hashlib
import zipfile
import tarfile
from datetime import datetime
import platform as plt

class LocalModelInstaller:
    def __init__(self, 
                 install_dir: str = None,
                 model_source: str = None,
                 enable_web_ui: bool = True,
                 enable_api: bool = True):
        
        # Détecter le système
        self.system = plt.system().lower()
        self.arch = plt.machine().lower()
        
        # Répertoires
        if install_dir is None:
            if self.system == "windows":
                install_dir = os.path.expanduser("~/AppData/Local/LLaMA-CyberSec")
            else:
                install_dir = os.path.expanduser("~/.llama-cybersec")
        
        self.install_dir = Path(install_dir)
        self.models_dir = self.install_dir / "models"
        self.config_dir = self.install_dir / "config"
        self.logs_dir = self.install_dir / "logs"
        self.web_dir = self.install_dir / "web"
        
        # Configuration
        self.model_source = model_source
        self.enable_web_ui = enable_web_ui
        self.enable_api = enable_api
        
        # Outils externes
        self.llama_cpp_path = None
        self.ollama_path = None
        
        self.setup_logging()
        self.create_directories()
        
    def setup_logging(self):
        """Configure le logging local"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / f'installation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Installation sur {self.system} {self.arch}")
    
    def create_directories(self):
        """Crée la structure de répertoires"""
        directories = [
            self.install_dir,
            self.models_dir,
            self.config_dir,
            self.logs_dir,
            self.web_dir,
            self.models_dir / "gguf",
            self.models_dir / "original",
            self.config_dir / "profiles"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Répertoires créés dans: {self.install_dir}")
    
    def check_system_requirements(self):
        """Vérifie les prérequis système"""
        self.logger.info("Vérification des prérequis système...")
        
        requirements = {
            "python": self.check_python(),
            "memory": self.check_memory(),
            "storage": self.check_storage(),
            "gpu": self.check_gpu()
        }
        
        # Afficher le résumé
        for req, status in requirements.items():
            icon = "✅" if status["status"] else "⚠️"
            self.logger.info(f"{icon} {req.upper()}: {status['message']}")
        
        return all(req["status"] for req in requirements.values())
    
    def check_python(self) -> Dict:
        """Vérifie Python"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return {"status": True, "message": f"Python {version.major}.{version.minor}"}
        else:
            return {"status": False, "message": f"Python {version.major}.{version.minor} < 3.8"}
    
    def check_memory(self) -> Dict:
        """Vérifie la mémoire RAM"""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb >= 8:
                return {"status": True, "message": f"{ram_gb:.1f} GB RAM"}
            else:
                return {"status": False, "message": f"{ram_gb:.1f} GB RAM < 8GB minimum"}
        except ImportError:
            return {"status": True, "message": "RAM non vérifiable"}
    
    def check_storage(self) -> Dict:
        """Vérifie l'espace disque"""
        try:
            import shutil
            free_gb = shutil.disk_usage(self.install_dir.parent)[2] / (1024**3)
            if free_gb >= 10:
                return {"status": True, "message": f"{free_gb:.1f} GB libre"}
            else:
                return {"status": False, "message": f"{free_gb:.1f} GB < 10GB requis"}
        except:
            return {"status": True, "message": "Espace disque non vérifiable"}
    
    def check_gpu(self) -> Dict:
        """Vérifie la disponibilité GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {"status": True, "message": f"{gpu_name} ({gpu_memory:.1f}GB)"}
            else:
                return {"status": True, "message": "CPU uniquement (OK)"}
        except ImportError:
            return {"status": True, "message": "PyTorch non installé"}
    
    def install_dependencies(self):
        """Installe les dépendances Python locales"""
        self.logger.info("Installation des dépendances...")
        
        # Dépendances minimales pour l'utilisation locale
        local_requirements = [
            "torch",
            "transformers",
            "requests",
            "flask",
            "flask-cors",
            "psutil",
            "pyyaml",
            "tqdm"
        ]
        
        try:
            for package in local_requirements:
                self.logger.info(f"Installation de {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True)
            
            self.logger.info("Dépendances installées avec succès")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur installation dépendances: {e}")
            return False
    
    def install_llama_cpp(self):
        """Installe llama.cpp pour l'inférence locale"""
        self.logger.info("Installation de llama.cpp...")
        
        llama_dir = self.install_dir / "llama.cpp"
        
        try:
            if not llama_dir.exists():
                # Cloner llama.cpp
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(llama_dir)
                ], check=True)
            
            # Compiler
            if self.system == "windows":
                # Sur Windows, utiliser les binaires pré-compilés ou MSVC
                self.logger.info("Compilation Windows - vérifiez les prérequis MSVC")
                subprocess.run([
                    "cmake", "-B", "build", "-S", ".", 
                    "-DLLAMA_CUDA=ON" if self.check_gpu()["status"] else ""
                ], cwd=llama_dir, check=True)
                subprocess.run([
                    "cmake", "--build", "build", "--config", "Release"
                ], cwd=llama_dir, check=True)
            else:
                # Linux/macOS
                subprocess.run([
                    "make", "-j4"
                ], cwd=llama_dir, check=True)
            
            self.llama_cpp_path = llama_dir
            self.logger.info("llama.cpp installé avec succès")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur installation llama.cpp: {e}")
            return False
    
    def install_ollama(self):
        """Installe Ollama (optionnel)"""
        self.logger.info("Installation d'Ollama...")
        
        try:
            if self.system == "windows":
                # Télécharger l'installateur Windows
                installer_url = "https://ollama.ai/download/OllamaSetup.exe"
                installer_path = self.install_dir / "OllamaSetup.exe"
                
                response = requests.get(installer_url)
                with open(installer_path, 'wb') as f:
                    f.write(response.content)
                
                self.logger.info(f"Installateur Ollama téléchargé: {installer_path}")
                self.logger.info("Veuillez exécuter l'installateur manuellement")
                
            else:
                # Linux/macOS
                subprocess.run([
                    "curl", "-fsSL", "https://ollama.ai/install.sh"
                ], check=True, stdout=subprocess.PIPE, shell=True)
            
            # Vérifier l'installation
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.ollama_path = "ollama"
                self.logger.info("Ollama installé avec succès")
                return True
            else:
                self.logger.warning("Ollama non disponible")
                return False
                
        except Exception as e:
            self.logger.warning(f"Installation Ollama échouée: {e}")
            return False
    
    def download_model(self, model_url: str = None):
        """Télécharge le modèle entraîné"""
        if not model_url and not self.model_source:
            self.logger.error("Aucune source de modèle spécifiée")
            return False
        
        source = model_url or self.model_source
        self.logger.info(f"Téléchargement du modèle: {source}")
        
        try:
            # Déterminer le type de source
            if source.startswith(("http://", "https://")):
                return self.download_from_url(source)
            elif source.startswith("s3://"):
                return self.download_from_s3(source)
            elif source.startswith("gs://"):
                return self.download_from_gcs(source)
            else:
                return self.copy_from_local(source)
                
        except Exception as e:
            self.logger.error(f"Erreur téléchargement modèle: {e}")
            return False
    
    def download_from_url(self, url: str) -> bool:
        """Télécharge depuis une URL HTTP"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Déterminer le nom de fichier
            filename = url.split('/')[-1]
            if not filename.endswith(('.gguf', '.bin', '.safetensors')):
                filename += '.gguf'
            
            model_path = self.models_dir / "gguf" / filename
            
            # Téléchargement avec barre de progression
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"\rTéléchargement: {progress:.1f}%", end='', flush=True)
            
            print()  # Nouvelle ligne après la barre de progression
            self.logger.info(f"Modèle téléchargé: {model_path}")
            
            # Vérifier l'intégrité si possible
            if self.verify_model_integrity(model_path):
                return True
            else:
                self.logger.error("Échec de vérification d'intégrité")
                return False
            
        except Exception as e:
            self.logger.error(f"Erreur téléchargement HTTP: {e}")
            return False
    
    def download_from_s3(self, s3_path: str) -> bool:
        """Télécharge depuis AWS S3"""
        try:
            import boto3
            # Implémenter le téléchargement S3
            # s3_path format: s3://bucket/path/to/model
            self.logger.info("Téléchargement S3 non implémenté")
            return False
        except ImportError:
            self.logger.error("boto3 requis pour S3")
            return False
    
    def download_from_gcs(self, gcs_path: str) -> bool:
        """Télécharge depuis Google Cloud Storage"""
        try:
            from google.cloud import storage
            # Implémenter le téléchargement GCS
            self.logger.info("Téléchargement GCS non implémenté")
            return False
        except ImportError:
            self.logger.error("google-cloud-storage requis pour GCS")
            return False
    
    def copy_from_local(self, source_path: str) -> bool:
        """Copie depuis un chemin local"""
        try:
            import shutil
            source = Path(source_path)
            
            if source.is_file():
                destination = self.models_dir / "gguf" / source.name
                shutil.copy2(source, destination)
                self.logger.info(f"Modèle copié: {destination}")
                return True
            else:
                self.logger.error(f"Source non trouvée: {source}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur copie locale: {e}")
            return False
    
    def verify_model_integrity(self, model_path: Path) -> bool:
        """Vérifie l'intégrité du modèle téléchargé"""
        try:
            # Vérification basique de la taille
            if model_path.stat().st_size < 1024 * 1024:  # < 1MB suspect
                self.logger.warning("Taille de modèle suspecte")
                return False
            
            # Vérification du format GGUF si applicable
            if model_path.suffix == '.gguf':
                with open(model_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'GGUF'):
                        self.logger.error("Format GGUF invalide")
                        return False
            
            self.logger.info("Intégrité du modèle vérifiée")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur vérification intégrité: {e}")
            return False
    
    def create_configuration(self):
        """Crée la configuration locale"""
        config = {
            "installation": {
                "date": datetime.now().isoformat(),
                "system": self.system,
                "architecture": self.arch,
                "install_dir": str(self.install_dir)
            },
            "model": {
                "path": str(self.models_dir / "gguf"),
                "type": "gguf",
                "quantization": "q4_0"
            },
            "inference": {
                "backend": "llama_cpp" if self.llama_cpp_path else "transformers",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "context_length": 2048
            },
            "web_ui": {
                "enabled": self.enable_web_ui,
                "host": "127.0.0.1",
                "port": 8080
            },
            "api": {
                "enabled": self.enable_api,
                "host": "127.0.0.1",
                "port": 8081
            }
        }
        
        config_path = self.config_dir / "local_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration sauvegardée: {config_path}")
        return config_path
    
    def create_launcher_scripts(self):
        """Crée les scripts de lancement"""
        # Script principal
        if self.system == "windows":
            launcher_script = self.install_dir / "launch.bat"
            script_content = f"""@echo off
cd /d "{self.install_dir}"
python local_deployment\\local_interface.py
pause
"""
        else:
            launcher_script = self.install_dir / "launch.sh"
            script_content = f"""#!/bin/bash
cd "{self.install_dir}"
python local_deployment/local_interface.py
"""
        
        with open(launcher_script, 'w') as f:
            f.write(script_content)
        
        if self.system != "windows":
            launcher_script.chmod(0o755)
        
        self.logger.info(f"Script de lancement créé: {launcher_script}")
    
    def setup_web_interface(self):
        """Configure l'interface web"""
        if not self.enable_web_ui:
            return
        
        # Créer les fichiers de l'interface web
        web_files = {
            "index.html": self.get_web_interface_html(),
            "style.css": self.get_web_interface_css(),
            "script.js": self.get_web_interface_js()
        }
        
        for filename, content in web_files.items():
            file_path = self.web_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        self.logger.info("Interface web configurée")
    
    def get_web_interface_html(self) -> str:
        """Retourne le code HTML de l'interface web"""
        return """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA CyberSec - Interface Locale</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ LLaMA CyberSec</h1>
            <p>Assistant IA spécialisé en cybersécurité</p>
        </header>
        
        <main>
            <div class="chat-container">
                <div id="messages" class="messages"></div>
                <div class="input-container">
                    <textarea id="prompt" placeholder="Posez votre question sur la cybersécurité..." rows="3"></textarea>
                    <button id="send" onclick="sendMessage()">Envoyer</button>
                </div>
            </div>
        </main>
        
        <aside class="sidebar">
            <h3>Exemples de questions</h3>
            <div class="examples">
                <button onclick="setPrompt('Créer une règle YARA pour détecter un ransomware')">Règle YARA</button>
                <button onclick="setPrompt('Analyser ce log Apache pour identifier une attaque')">Analyse de logs</button>
                <button onclick="setPrompt('Expliquer les techniques de lateral movement')">Techniques APT</button>
                <button onclick="setPrompt('Corriger ce code Python vulnérable')">Sécurité du code</button>
            </div>
            
            <h3>Configuration</h3>
            <div class="config">
                <label>Température: <span id="temp-value">0.7</span></label>
                <input type="range" id="temperature" min="0.1" max="1.0" step="0.1" value="0.7">
                
                <label>Longueur max: <span id="length-value">512</span></label>
                <input type="range" id="max-length" min="100" max="1000" step="50" value="512">
            </div>
        </aside>
    </div>
    
    <script src="script.js"></script>
</body>
</html>"""
    
    def get_web_interface_css(self) -> str:
        """Retourne le CSS de l'interface web"""
        return """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: #333;
    min-height: 100vh;
}

.container {
    display: grid;
    grid-template-areas: 
        "header header"
        "main sidebar";
    grid-template-columns: 1fr 300px;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    box-shadow: 0 0 20px rgba(0,0,0,0.1);
}

header {
    grid-area: header;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    text-align: center;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

main {
    grid-area: main;
    padding: 2rem;
    display: flex;
    flex-direction: column;
}

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    max-height: 70vh;
}

.messages {
    flex: 1;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    overflow-y: auto;
    background: #fafafa;
}

.message {
    margin-bottom: 1rem;
    padding: 1rem;
    border-radius: 8px;
}

.message.user {
    background: #e3f2fd;
    margin-left: 2rem;
}

.message.assistant {
    background: #f3e5f5;
    margin-right: 2rem;
}

.input-container {
    display: flex;
    gap: 1rem;
}

#prompt {
    flex: 1;
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    resize: vertical;
    font-family: inherit;
}

#send {
    background: #667eea;
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
}

#send:hover {
    background: #5a67d8;
}

.sidebar {
    grid-area: sidebar;
    background: #f8f9fa;
    padding: 2rem;
    border-left: 1px solid #ddd;
}

.sidebar h3 {
    margin-bottom: 1rem;
    color: #495057;
}

.examples button {
    display: block;
    width: 100%;
    margin-bottom: 0.5rem;
    padding: 0.8rem;
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    cursor: pointer;
    text-align: left;
    font-size: 0.9rem;
}

.examples button:hover {
    background: #e9ecef;
}

.config {
    margin-top: 2rem;
}

.config label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

.config input[type="range"] {
    width: 100%;
    margin-bottom: 1rem;
}

@media (max-width: 768px) {
    .container {
        grid-template-areas: 
            "header"
            "main"
            "sidebar";
        grid-template-columns: 1fr;
    }
    
    .sidebar {
        border-left: none;
        border-top: 1px solid #ddd;
    }
}"""
    
    def get_web_interface_js(self) -> str:
        """Retourne le JavaScript de l'interface web"""
        return """let chatHistory = [];

function setPrompt(text) {
    document.getElementById('prompt').value = text;
}

function updateConfig() {
    const temperature = document.getElementById('temperature').value;
    const maxLength = document.getElementById('max-length').value;
    
    document.getElementById('temp-value').textContent = temperature;
    document.getElementById('length-value').textContent = maxLength;
}

// Mise à jour des valeurs affichées
document.getElementById('temperature').addEventListener('input', updateConfig);
document.getElementById('max-length').addEventListener('input', updateConfig);

async function sendMessage() {
    const promptElement = document.getElementById('prompt');
    const messagesElement = document.getElementById('messages');
    const sendButton = document.getElementById('send');
    
    const prompt = promptElement.value.trim();
    if (!prompt) return;
    
    // Désactiver le bouton d'envoi
    sendButton.disabled = true;
    sendButton.textContent = 'Traitement...';
    
    // Ajouter le message utilisateur
    addMessage('user', prompt);
    promptElement.value = '';
    
    try {
        // Paramètres de configuration
        const temperature = parseFloat(document.getElementById('temperature').value);
        const maxLength = parseInt(document.getElementById('max-length').value);
        
        // Envoyer la requête à l'API locale
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                temperature: temperature,
                max_tokens: maxLength,
                history: chatHistory
            })
        });
        
        if (!response.ok) {
            throw new Error(`Erreur API: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Ajouter la réponse
        addMessage('assistant', data.response);
        
        // Mettre à jour l'historique
        chatHistory.push({user: prompt, assistant: data.response});
        
    } catch (error) {
        console.error('Erreur:', error);
        addMessage('assistant', '❌ Erreur: Impossible de traiter votre demande. Vérifiez que le service local est démarré.');
    } finally {
        // Réactiver le bouton
        sendButton.disabled = false;
        sendButton.textContent = 'Envoyer';
    }
}

function addMessage(type, text) {
    const messagesElement = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    if (type === 'user') {
        messageDiv.innerHTML = `<strong>Vous:</strong><br>${escapeHtml(text)}`;
    } else {
        messageDiv.innerHTML = `<strong>🛡️ LLaMA CyberSec:</strong><br>${formatResponse(text)}`;
    }
    
    messagesElement.appendChild(messageDiv);
    messagesElement.scrollTop = messagesElement.scrollHeight;
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatResponse(text) {
    // Formatage basique pour améliorer l'affichage
    return text
        .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
        .replace(/\\n/g, '<br>');
}

// Gérer l'envoi avec Entrée
document.getElementById('prompt').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Message d'accueil
document.addEventListener('DOMContentLoaded', function() {
    addMessage('assistant', '👋 Bonjour ! Je suis votre assistant IA spécialisé en cybersécurité. Je peux vous aider avec:\\n\\n• Création de règles de détection (Sigma, YARA, Snort)\\n• Analyse de logs de sécurité\\n• Audit et correction de code\\n• Explications sur les techniques d\\'attaque\\n• Conseils de sécurisation\\n\\nQue puis-je faire pour vous ?');
});"""
    
    def run_full_installation(self):
        """Exécute l'installation complète"""
        self.logger.info("=== DÉBUT DE L'INSTALLATION LOCALE ===")
        
        try:
            # 1. Vérifications système
            if not self.check_system_requirements():
                self.logger.warning("Certains prérequis ne sont pas satisfaits")
                response = input("Continuer malgré tout ? (y/N): ")
                if response.lower() != 'y':
                    return False
            
            # 2. Installation des dépendances
            if not self.install_dependencies():
                self.logger.error("Échec installation dépendances")
                return False
            
            # 3. Installation des outils d'inférence
            if not self.install_llama_cpp():
                self.logger.warning("llama.cpp non installé - utilisation de transformers")
            
            self.install_ollama()  # Optionnel
            
            # 4. Téléchargement du modèle
            if self.model_source:
                if not self.download_model():
                    self.logger.error("Échec téléchargement modèle")
                    return False
            else:
                self.logger.warning("Aucun modèle spécifié - installation des outils seulement")
            
            # 5. Configuration
            self.create_configuration()
            self.create_launcher_scripts()
            
            # 6. Interface web
            if self.enable_web_ui:
                self.setup_web_interface()
            
            self.logger.info("=== INSTALLATION TERMINÉE AVEC SUCCÈS ===")
            self.show_completion_summary()
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'installation: {e}")
            return False
    
    def show_completion_summary(self):
        """Affiche le résumé de fin d'installation"""
        print(f"\n{'='*60}")
        print("🎉 LLAMA CYBERSEC INSTALLÉ AVEC SUCCÈS!")
        print(f"{'='*60}")
        print(f"\n📁 Répertoire d'installation: {self.install_dir}")
        print(f"🔧 Configuration: {self.config_dir / 'local_config.json'}")
        print(f"📊 Logs: {self.logs_dir}")
        
        if self.enable_web_ui:
            print(f"🌐 Interface web: http://localhost:8080")
        
        if self.enable_api:
            print(f"🔌 API REST: http://localhost:8081")
        
        print(f"\n🚀 DÉMARRAGE:")
        if self.system == "windows":
            print(f"  Double-cliquez sur: {self.install_dir / 'launch.bat'}")
        else:
            print(f"  Exécutez: {self.install_dir / 'launch.sh'}")
        
        print(f"\n💡 AIDE:")
        print(f"  • Documentation: {self.install_dir / 'docs'}")
        print(f"  • Configuration: {self.config_dir}")
        print(f"  • Logs de debug: {self.logs_dir}")


def main():
    parser = argparse.ArgumentParser(description='Installation locale LLaMA CyberSec')
    parser.add_argument('--install-dir', help='Répertoire d\'installation')
    parser.add_argument('--model-source', help='Source du modèle (URL, chemin local, etc.)')
    parser.add_argument('--no-web-ui', action='store_true', help='Désactiver l\'interface web')
    parser.add_argument('--no-api', action='store_true', help='Désactiver l\'API REST')
    parser.add_argument('--check-only', action='store_true', help='Vérifier les prérequis seulement')
    
    args = parser.parse_args()
    
    # Initialiser l'installateur
    installer = LocalModelInstaller(
        install_dir=args.install_dir,
        model_source=args.model_source,
        enable_web_ui=not args.no_web_ui,
        enable_api=not args.no_api
    )
    
    if args.check_only:
        installer.check_system_requirements()
        return
    
    # Lancer l'installation complète
    success = installer.run_full_installation()
    
    if success:
        print("\n✅ Installation réussie!")
        print("🚀 Vous pouvez maintenant utiliser LLaMA CyberSec en local")
    else:
        print("\n❌ Installation échouée")
        print("📋 Consultez les logs pour plus de détails")
        sys.exit(1)


if __name__ == "__main__":
    main()