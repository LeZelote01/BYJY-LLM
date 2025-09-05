#!/usr/bin/env python3
"""
Interface locale pour LLaMA-3-8B Cybersécurité
Serveur web et API REST pour l'utilisation du modèle en local
Support: CPU optimisé, GPU léger, streaming, session management
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime
import uuid

from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import threading
import queue
import time

class LocalLLaMAInterface:
    def __init__(self, config_path: str = None):
        # Configuration
        self.config_path = config_path or self.find_config()
        self.config = self.load_config()
        
        # Paths
        self.install_dir = Path(self.config["installation"]["install_dir"])
        self.models_dir = self.install_dir / "models"
        self.logs_dir = self.install_dir / "logs"
        
        # Setup logging
        self.setup_logging()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False
        
        # Session management
        self.sessions = {}
        self.max_sessions = 10
        
        # Flask app
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        CORS(self.app)
        
        # Setup routes
        self.setup_routes()
        
        # Load model in background
        self.model_load_thread = threading.Thread(target=self.load_model_async)
        self.model_load_thread.daemon = True
        self.model_load_thread.start()
    
    def find_config(self) -> str:
        """Trouve le fichier de configuration"""
        possible_paths = [
            "./config/local_config.json",
            os.path.expanduser("~/.llama-cybersec/config/local_config.json"),
            os.path.expanduser("~/AppData/Local/LLaMA-CyberSec/config/local_config.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration locale non trouvée")
    
    def load_config(self) -> Dict:
        """Charge la configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def setup_logging(self):
        """Configure le logging"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / f'interface_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_async(self):
        """Charge le modèle en arrière-plan"""
        try:
            self.logger.info("Chargement du modèle...")
            
            # Déterminer le chemin du modèle
            model_path = self.find_model_file()
            if not model_path:
                self.logger.error("Aucun modèle trouvé")
                return
            
            # Charger selon le format
            if model_path.suffix == '.gguf':
                self.load_gguf_model(model_path)
            else:
                self.load_transformers_model(model_path)
            
            self.model_loaded = True
            self.logger.info("Modèle chargé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            self.model_loaded = False
    
    def find_model_file(self) -> Optional[Path]:
        """Trouve le fichier de modèle"""
        model_dir = Path(self.config["model"]["path"])
        
        # Chercher des fichiers de modèle
        extensions = ['.gguf', '.bin', '.safetensors']
        
        for ext in extensions:
            files = list(model_dir.glob(f"*{ext}"))
            if files:
                return files[0]  # Prendre le premier trouvé
        
        return None
    
    def load_gguf_model(self, model_path: Path):
        """Charge un modèle GGUF avec llama.cpp"""
        try:
            # Importer llama-cpp-python si disponible
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config["inference"]["context_length"],
                n_threads=os.cpu_count(),
                verbose=False
            )
            self.logger.info(f"Modèle GGUF chargé: {model_path}")
            
        except ImportError:
            self.logger.error("llama-cpp-python non installé")
            raise
        except Exception as e:
            self.logger.error(f"Erreur chargement GGUF: {e}")
            raise
    
    def load_transformers_model(self, model_path: Path):
        """Charge un modèle avec Transformers"""
        try:
            # Charger tokenizer et modèle
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Créer pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info(f"Modèle Transformers chargé: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement Transformers: {e}")
            raise
    
    def format_prompt(self, instruction: str) -> str:
        """Formate le prompt pour l'inférence"""
        return f"""### Instruction:
{instruction}

### Response:
"""
    
    def generate_response(self, prompt: str, session_id: str = None, **kwargs) -> str:
        """Génère une réponse"""
        if not self.model_loaded:
            return "❌ Modèle en cours de chargement, veuillez patienter..."
        
        try:
            # Paramètres de génération
            temperature = kwargs.get('temperature', self.config["inference"]["temperature"])
            max_tokens = kwargs.get('max_tokens', self.config["inference"]["max_tokens"])
            top_p = kwargs.get('top_p', self.config["inference"]["top_p"])
            
            # Contexte de session
            full_prompt = self.build_context_prompt(prompt, session_id)
            
            # Génération selon le type de modèle
            if hasattr(self.model, 'create_completion'):  # llama.cpp
                response = self.model.create_completion(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["### Instruction:", "### Response:"]
                )
                text = response['choices'][0]['text'].strip()
            
            elif self.pipeline:  # Transformers
                outputs = self.pipeline(
                    full_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = outputs[0]['generated_text']
                # Extraire seulement la réponse
                if "### Response:" in generated_text:
                    text = generated_text.split("### Response:")[-1].strip()
                else:
                    text = generated_text[len(full_prompt):].strip()
            
            else:
                return "❌ Aucun modèle disponible"
            
            # Mettre à jour la session
            if session_id:
                self.update_session(session_id, prompt, text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Erreur génération: {e}")
            return f"❌ Erreur lors de la génération: {str(e)}"
    
    def build_context_prompt(self, current_prompt: str, session_id: str = None) -> str:
        """Construit le prompt avec contexte de session"""
        base_prompt = self.format_prompt(current_prompt)
        
        if not session_id or session_id not in self.sessions:
            return base_prompt
        
        # Ajouter le contexte de la session
        session = self.sessions[session_id]
        context_parts = []
        
        # Prendre les N derniers échanges
        recent_history = session["history"][-3:]  # 3 derniers échanges
        
        for exchange in recent_history:
            context_parts.append(f"### Instruction:\n{exchange['user']}")
            context_parts.append(f"### Response:\n{exchange['assistant']}")
        
        if context_parts:
            context = "\n\n".join(context_parts)
            return f"{context}\n\n{base_prompt}"
        
        return base_prompt
    
    def get_or_create_session(self, session_id: str = None) -> str:
        """Obtient ou crée une session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            # Nettoyer les anciennes sessions si nécessaire
            if len(self.sessions) >= self.max_sessions:
                oldest_session = min(
                    self.sessions.keys(),
                    key=lambda k: self.sessions[k]["created_at"]
                )
                del self.sessions[oldest_session]
            
            # Créer nouvelle session
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "history": [],
                "last_activity": datetime.now()
            }
        
        return session_id
    
    def update_session(self, session_id: str, user_input: str, assistant_response: str):
        """Met à jour une session avec un nouvel échange"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                "user": user_input,
                "assistant": assistant_response,
                "timestamp": datetime.now().isoformat()
            })
            self.sessions[session_id]["last_activity"] = datetime.now()
            
            # Limiter l'historique
            if len(self.sessions[session_id]["history"]) > 20:
                self.sessions[session_id]["history"] = self.sessions[session_id]["history"][-20:]
    
    def setup_routes(self):
        """Configure les routes Flask"""
        
        @self.app.route('/')
        def index():
            """Page d'accueil avec interface web"""
            if not self.config["web_ui"]["enabled"]:
                return jsonify({"error": "Interface web désactivée"}), 403
            
            # Charger l'interface web
            web_dir = self.install_dir / "web"
            
            try:
                with open(web_dir / "index.html", 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return self.get_simple_interface()
        
        @self.app.route('/api/status')
        def status():
            """Status de l'API"""
            return jsonify({
                "status": "online",
                "model_loaded": self.model_loaded,
                "sessions": len(self.sessions),
                "config": {
                    "web_ui": self.config["web_ui"]["enabled"],
                    "api": self.config["api"]["enabled"]
                }
            })
        
        @self.app.route('/api/generate', methods=['POST'])
        def generate():
            """Génération de texte"""
            if not self.config["api"]["enabled"]:
                return jsonify({"error": "API désactivée"}), 403
            
            try:
                data = request.get_json()
                
                if not data or 'prompt' not in data:
                    return jsonify({"error": "Prompt requis"}), 400
                
                prompt = data['prompt']
                session_id = data.get('session_id')
                
                # Créer ou récupérer session
                if session_id:
                    session_id = self.get_or_create_session(session_id)
                
                # Générer réponse
                response = self.generate_response(
                    prompt,
                    session_id=session_id,
                    temperature=data.get('temperature', 0.7),
                    max_tokens=data.get('max_tokens', 512),
                    top_p=data.get('top_p', 0.9)
                )
                
                return jsonify({
                    "response": response,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Erreur API generate: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/session/<session_id>')
        def get_session(session_id):
            """Récupère l'historique d'une session"""
            if session_id not in self.sessions:
                return jsonify({"error": "Session non trouvée"}), 404
            
            session = self.sessions[session_id]
            return jsonify({
                "session_id": session_id,
                "created_at": session["created_at"].isoformat(),
                "last_activity": session["last_activity"].isoformat(),
                "history": session["history"]
            })
        
        @self.app.route('/api/session/<session_id>', methods=['DELETE'])
        def delete_session(session_id):
            """Supprime une session"""
            if session_id in self.sessions:
                del self.sessions[session_id]
                return jsonify({"message": "Session supprimée"})
            else:
                return jsonify({"error": "Session non trouvée"}), 404
        
        @self.app.route('/api/models')
        def list_models():
            """Liste les modèles disponibles"""
            models_info = []
            
            if self.model_loaded:
                model_file = self.find_model_file()
                if model_file:
                    models_info.append({
                        "name": model_file.stem,
                        "path": str(model_file),
                        "format": model_file.suffix[1:],
                        "loaded": True
                    })
            
            return jsonify({"models": models_info})
        
        @self.app.route('/health')
        def health():
            """Health check"""
            return jsonify({
                "status": "healthy",
                "model_loaded": self.model_loaded,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_simple_interface(self) -> str:
        """Interface web simple intégrée"""
        return """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA CyberSec - Interface Simple</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 20px; }
        .input-container { display: flex; gap: 10px; }
        #prompt { flex: 1; padding: 10px; }
        #send { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .message { margin-bottom: 15px; }
        .user { background: #e3f2fd; padding: 10px; border-radius: 5px; }
        .assistant { background: #f3e5f5; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ LLaMA CyberSec</h1>
        <p>Assistant IA local spécialisé en cybersécurité</p>
    </div>
    
    <div id="chat" class="chat-container"></div>
    
    <div class="input-container">
        <textarea id="prompt" placeholder="Posez votre question..." rows="3"></textarea>
        <button id="send" onclick="sendMessage()">Envoyer</button>
    </div>
    
    <script>
        let sessionId = null;
        
        function addMessage(type, text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = `<strong>${type === 'user' ? 'Vous' : '🛡️ Assistant'}:</strong><br>${text.replace(/\\n/g, '<br>')}`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        async function sendMessage() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;
            
            addMessage('user', prompt);
            document.getElementById('prompt').value = '';
            document.getElementById('send').disabled = true;
            document.getElementById('send').textContent = 'Traitement...';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage('assistant', data.response);
                    sessionId = data.session_id;
                } else {
                    addMessage('assistant', `Erreur: ${data.error}`);
                }
            } catch (error) {
                addMessage('assistant', `Erreur de connexion: ${error.message}`);
            } finally {
                document.getElementById('send').disabled = false;
                document.getElementById('send').textContent = 'Envoyer';
            }
        }
        
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Message d'accueil
        addMessage('assistant', 'Bonjour ! Je suis votre assistant IA spécialisé en cybersécurité. Comment puis-je vous aider ?');
    </script>
</body>
</html>"""
    
    def run(self):
        """Lance le serveur"""
        host = self.config["web_ui"]["host"]
        port = self.config["web_ui"]["port"]
        
        self.logger.info(f"Démarrage du serveur sur http://{host}:{port}")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=False,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Erreur serveur: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interface locale LLaMA CyberSec')
    parser.add_argument('--config', help='Chemin vers le fichier de configuration')
    parser.add_argument('--host', default='127.0.0.1', help='Adresse d\'écoute')
    parser.add_argument('--port', type=int, default=8080, help='Port d\'écoute')
    
    args = parser.parse_args()
    
    try:
        # Initialiser l'interface
        interface = LocalLLaMAInterface(config_path=args.config)
        
        # Surcharger config si arguments fournis
        if args.host != '127.0.0.1':
            interface.config["web_ui"]["host"] = args.host
        if args.port != 8080:
            interface.config["web_ui"]["port"] = args.port
        
        # Lancer le serveur
        interface.run()
        
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        print("💡 Utilisez d'abord local_installer.py pour installer le système")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Arrêt du serveur...")
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()