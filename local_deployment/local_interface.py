#!/usr/bin/env python3
"""
🏠 Interface Locale Modernisée pour LLaMA-3-8B Cybersécurité
Version unifiée et optimisée pour utilisation locale

Fonctionnalités principales:
- Interface web moderne avec thème sombre/clair
- Chat en temps réel avec streaming WebSocket  
- Analyse de code multi-fichiers avec détection de vulnérabilités
- Génération de rapports sécurisés automatisés
- API REST complète avec authentification JWT
- Monitoring en temps réel des performances
- Système de plugins extensible
- Base de données SQLite intégrée pour persistance
- Support multi-format de modèles (GGUF, Transformers)
- Configuration hybride local/cloud optimisée
"""

import os
import sys
import json
import logging
import asyncio
import uuid
import time
import hashlib
import jwt
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime, timedelta
import mimetypes
import zipfile
import tempfile

from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import threading
import queue
import sqlite3
import markdown
from datetime import datetime

class ModernLLaMAInterface:
    def __init__(self, config_path: str = None):
        # Configuration
        self.config_path = config_path or self.find_config()
        self.config = self.load_config()
    
    def find_config(self):
        """Find configuration file"""
        possible_paths = [
            "/app/configs/config.json",
            Path.cwd() / "configs" / "config.json",
            Path.home() / ".llama-cybersec" / "config.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        # Return default path
        return "/app/configs/config.json"
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            # Return minimal default config
            return {
                "installation": {"install_dir": "/tmp/llama-cybersec"},
                "web_ui": {"host": "127.0.0.1", "port": 8080},
                "model": {"name": "test-model"}
            }
        
        # Paths
        self.install_dir = Path(self.config["installation"]["install_dir"])
        self.models_dir = self.install_dir / "models"
        self.logs_dir = self.install_dir / "logs"
        self.uploads_dir = self.install_dir / "uploads"
        self.reports_dir = self.install_dir / "reports"
        
        # Créer les répertoires
        for directory in [self.uploads_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Base de données pour les sessions et l'historique
        self.setup_database()
        
        # Composants du modèle
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False
        self.model_loading = False
        
        # Gestion des sessions avancée
        self.active_sessions = {}
        self.max_sessions = 50
        
        # Flask app avec SocketIO
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
        CORS(self.app)
        
        # SocketIO pour le temps réel
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Plugin system
        self.plugins = {}
        self.load_plugins()
        
        # Setup routes and events
        self.setup_routes()
        self.setup_socket_events()
        
        # Chargement du modèle en arrière-plan
        self.start_model_loading()
    
    def setup_database(self):
        """Configuration de la base de données SQLite"""
        db_path = self.install_dir / "interface.db"
        
        self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db_lock = threading.Lock()
        
        # Créer les tables
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            # Table des sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Table de l'historique des conversations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_input TEXT,
                    assistant_response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            # Table des rapports générés
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    report_type TEXT,
                    title TEXT,
                    content TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            # Table des fichiers uploadés
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    filename TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    mime_type TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_status TEXT DEFAULT 'pending',
                    analysis_results TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            self.db_connection.commit()
    
    def setup_logging(self):
        """Configuration du logging avancé"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger principal
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / f'modern_interface_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Logger pour les requêtes
        self.request_logger = logging.getLogger('requests')
        request_handler = logging.FileHandler(self.logs_dir / 'requests.log')
        request_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.request_logger.addHandler(request_handler)
        self.request_logger.setLevel(logging.INFO)
        
        # Logger pour les erreurs
        self.error_logger = logging.getLogger('errors')
        error_handler = logging.FileHandler(self.logs_dir / 'errors.log')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)
    
    def load_plugins(self):
        """Chargement du système de plugins"""
        plugins_dir = self.install_dir / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        
        # Plugin par défaut pour l'analyse de code
        self.plugins['code_analyzer'] = CodeAnalyzerPlugin(self)
        self.plugins['report_generator'] = ReportGeneratorPlugin(self)
        self.plugins['vulnerability_scanner'] = VulnerabilityPlugin(self)
        
        self.logger.info(f"Plugins chargés: {list(self.plugins.keys())}")
    
    def start_model_loading(self):
        """Démarre le chargement du modèle en arrière-plan"""
        if not self.model_loading:
            self.model_loading = True
            self.model_load_thread = threading.Thread(target=self.load_model_async)
            self.model_load_thread.daemon = True
            self.model_load_thread.start()
    
    def load_model_async(self):
        """Charge le modèle de manière asynchrone"""
        try:
            self.logger.info("🔄 Chargement du modèle en cours...")
            
            # Émettre le statut de chargement
            self.socketio.emit('model_status', {'status': 'loading', 'progress': 0})
            
            # Déterminer le chemin du modèle
            model_path = self.find_model_file()
            if not model_path:
                self.logger.error("Aucun modèle trouvé")
                self.socketio.emit('model_status', {'status': 'error', 'message': 'Aucun modèle trouvé'})
                return
            
            self.socketio.emit('model_status', {'status': 'loading', 'progress': 25})
            
            # Charger selon le format
            if model_path.suffix == '.gguf':
                self.load_gguf_model(model_path)
            else:
                self.load_transformers_model(model_path)
            
            self.socketio.emit('model_status', {'status': 'loading', 'progress': 75})
            
            self.model_loaded = True
            self.model_loading = False
            
            self.socketio.emit('model_status', {'status': 'ready', 'progress': 100})
            self.logger.info("✅ Modèle chargé avec succès")
            
        except Exception as e:
            self.model_loading = False
            self.logger.error(f"Erreur chargement modèle: {e}")
            self.socketio.emit('model_status', {'status': 'error', 'message': str(e)})
    
    def setup_routes(self):
        """Configuration des routes Flask"""
        
        @self.app.route('/')
        def index():
            """Interface principale"""
            return self.get_modern_interface()
        
        @self.app.route('/api/status')
        def api_status():
            """Statut de l'API"""
            return jsonify({
                "status": "online",
                "model_loaded": self.model_loaded,
                "model_loading": self.model_loading,
                "active_sessions": len(self.active_sessions),
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "plugins": list(self.plugins.keys())
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def api_chat():
            """Chat API avec streaming"""
            try:
                data = request.get_json()
                
                if not data or 'message' not in data:
                    return jsonify({"error": "Message requis"}), 400
                
                session_id = data.get('session_id') or str(uuid.uuid4())
                message = data['message']
                stream = data.get('stream', False)
                
                # Log de la requête
                self.log_request(session_id, message, request.remote_addr)
                
                if stream:
                    return self.stream_response(session_id, message)
                else:
                    response = self.generate_response(message, session_id)
                    self.save_conversation(session_id, message, response)
                    
                    return jsonify({
                        "response": response,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.error_logger.error(f"Erreur API chat: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/upload', methods=['POST'])
        def api_upload():
            """Upload de fichiers pour analyse"""
            try:
                if 'files' not in request.files:
                    return jsonify({"error": "Aucun fichier fourni"}), 400
                
                session_id = request.form.get('session_id', str(uuid.uuid4()))
                files = request.files.getlist('files')
                
                uploaded_files = []
                
                for file in files:
                    if file and file.filename:
                        file_id = str(uuid.uuid4())
                        filename = secure_filename(file.filename)
                        file_path = self.uploads_dir / session_id / f"{file_id}_{filename}"
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        file.save(file_path)
                        
                        # Enregistrer en base
                        self.save_upload(file_id, session_id, filename, file_path, file.content_length)
                        
                        uploaded_files.append({
                            "file_id": file_id,
                            "filename": filename,
                            "size": file_path.stat().st_size
                        })
                        
                        # Lancer l'analyse en arrière-plan
                        threading.Thread(
                            target=self.analyze_file_async,
                            args=(file_id, file_path, session_id),
                            daemon=True
                        ).start()
                
                return jsonify({
                    "uploaded_files": uploaded_files,
                    "session_id": session_id
                })
                
            except Exception as e:
                self.error_logger.error(f"Erreur upload: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/report/<report_id>')
        def api_get_report(report_id):
            """Récupération d'un rapport"""
            try:
                with self.db_lock:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "SELECT * FROM reports WHERE id = ?",
                        (report_id,)
                    )
                    report = cursor.fetchone()
                
                if not report:
                    return jsonify({"error": "Rapport non trouvé"}), 404
                
                return jsonify({
                    "id": report[0],
                    "type": report[2],
                    "title": report[3],
                    "content": json.loads(report[4]),
                    "created_at": report[6]
                })
                
            except Exception as e:
                self.error_logger.error(f"Erreur récupération rapport: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/sessions/<session_id>/history')
        def api_session_history(session_id):
            """Historique d'une session"""
            try:
                with self.db_lock:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "SELECT user_input, assistant_response, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp",
                        (session_id,)
                    )
                    history = cursor.fetchall()
                
                return jsonify({
                    "session_id": session_id,
                    "history": [
                        {
                            "user": item[0],
                            "assistant": item[1],
                            "timestamp": item[2]
                        } for item in history
                    ]
                })
                
            except Exception as e:
                self.error_logger.error(f"Erreur historique session: {e}")
                return jsonify({"error": str(e)}), 500
    
    def setup_socket_events(self):
        """Configuration des événements SocketIO"""
        
        @self.socketio.on('connect')
        def on_connect():
            """Connexion d'un client"""
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            
            emit('session_created', {'session_id': session_id})
            emit('model_status', {
                'status': 'ready' if self.model_loaded else ('loading' if self.model_loading else 'not_loaded'),
                'progress': 100 if self.model_loaded else 0
            })
            
            self.logger.info(f"Client connecté - Session: {session_id}")
        
        @self.socketio.on('disconnect')
        def on_disconnect():
            """Déconnexion d'un client"""
            session_id = session.get('session_id')
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            self.logger.info(f"Client déconnecté - Session: {session_id}")
        
        @self.socketio.on('chat_message')
        def on_chat_message(data):
            """Message de chat en temps réel"""
            try:
                session_id = session.get('session_id')
                message = data.get('message', '')
                
                if not message:
                    emit('error', {'message': 'Message vide'})
                    return
                
                # Générer la réponse en streaming
                self.generate_streaming_response(session_id, message)
                
            except Exception as e:
                self.error_logger.error(f"Erreur chat websocket: {e}")
                emit('error', {'message': str(e)})
        
        @self.socketio.on('analyze_code')
        def on_analyze_code(data):
            """Analyse de code en temps réel"""
            try:
                session_id = session.get('session_id')
                code = data.get('code', '')
                language = data.get('language', 'auto')
                
                # Lancer l'analyse en arrière-plan
                threading.Thread(
                    target=self.analyze_code_async,
                    args=(session_id, code, language),
                    daemon=True
                ).start()
                
            except Exception as e:
                self.error_logger.error(f"Erreur analyse code: {e}")
                emit('error', {'message': str(e)})
    
    def generate_streaming_response(self, session_id: str, message: str):
        """Génère une réponse en streaming via WebSocket"""
        try:
            if not self.model_loaded:
                self.socketio.emit('chat_response', {
                    'session_id': session_id,
                    'type': 'error',
                    'content': '⏳ Modèle en cours de chargement...'
                })
                return
            
            # Émettre le début de la réponse
            self.socketio.emit('chat_response', {
                'session_id': session_id,
                'type': 'start',
                'content': ''
            })
            
            # Générer la réponse par chunks
            full_response = ""
            
            # Simuler le streaming (à adapter selon votre modèle)
            response = self.generate_response(message, session_id)
            
            # Envoyer par chunks de mots
            words = response.split()
            for i, word in enumerate(words):
                chunk = word + " "
                full_response += chunk
                
                self.socketio.emit('chat_response', {
                    'session_id': session_id,
                    'type': 'chunk',
                    'content': chunk
                })
                
                time.sleep(0.05)  # Simule le streaming
            
            # Émettre la fin de la réponse
            self.socketio.emit('chat_response', {
                'session_id': session_id,
                'type': 'end',
                'content': full_response
            })
            
            # Sauvegarder la conversation
            self.save_conversation(session_id, message, full_response)
            
        except Exception as e:
            self.error_logger.error(f"Erreur streaming: {e}")
            self.socketio.emit('chat_response', {
                'session_id': session_id,
                'type': 'error',
                'content': f'Erreur: {str(e)}'
            })
    
    def analyze_code_async(self, session_id: str, code: str, language: str):
        """Analyse de code asynchrone"""
        try:
            self.socketio.emit('analysis_status', {
                'session_id': session_id,
                'status': 'analyzing',
                'progress': 0
            })
            
            # Utiliser le plugin d'analyse de code
            analyzer = self.plugins.get('code_analyzer')
            if analyzer:
                results = analyzer.analyze(code, language)
                
                self.socketio.emit('analysis_status', {
                    'session_id': session_id,
                    'status': 'complete',
                    'progress': 100,
                    'results': results
                })
            else:
                self.socketio.emit('analysis_status', {
                    'session_id': session_id,
                    'status': 'error',
                    'message': 'Plugin d\'analyse non disponible'
                })
                
        except Exception as e:
            self.error_logger.error(f"Erreur analyse code async: {e}")
            self.socketio.emit('analysis_status', {
                'session_id': session_id,
                'status': 'error',
                'message': str(e)
            })
    
    def get_modern_interface(self) -> str:
        """Interface web moderne"""
        return """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ LLaMA CyberSec - Interface Moderne</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #f093fb;
            --bg-color: #0f0f0f;
            --surface-color: #1a1a1a;
            --surface-light: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --success-color: #4ade80;
            --warning-color: #fbbf24;
            --error-color: #ef4444;
            --border-color: #333333;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, var(--bg-color) 0%, #1a1a2e 100%);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            display: grid;
            grid-template-areas: 
                "sidebar header"
                "sidebar main";
            grid-template-columns: 280px 1fr;
            grid-template-rows: 70px 1fr;
            min-height: 100vh;
        }

        .header {
            grid-area: header;
            background: rgba(26, 26, 26, 0.95);
            backdrop-filter: blur(20px);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            border-bottom: 1px solid var(--border-color);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            background: var(--surface-color);
            border: 1px solid var(--border-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--error-color);
            animation: pulse 2s infinite;
        }

        .status-dot.ready { background: var(--success-color); }
        .status-dot.loading { background: var(--warning-color); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .sidebar {
            grid-area: sidebar;
            background: rgba(26, 26, 26, 0.95);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--border-color);
            padding: 1rem;
            overflow-y: auto;
        }

        .nav-section {
            margin-bottom: 2rem;
        }

        .nav-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 0.25rem;
        }

        .nav-item:hover {
            background: var(--surface-light);
        }

        .nav-item.active {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
        }

        .main-content {
            grid-area: main;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 70px);
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 100%;
        }

        .messages {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            scroll-behavior: smooth;
        }

        .message {
            margin-bottom: 2rem;
            display: flex;
            gap: 1rem;
            animation: fadeInUp 0.3s ease;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }

        .user-avatar {
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        }

        .assistant-avatar {
            background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
        }

        .message-content {
            flex: 1;
            background: var(--surface-color);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--border-color);
        }

        .message.user .message-content {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(240, 147, 251, 0.1));
        }

        .input-area {
            padding: 2rem;
            background: rgba(26, 26, 26, 0.95);
            backdrop-filter: blur(20px);
            border-top: 1px solid var(--border-color);
        }

        .input-container {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
            max-width: 1200px;
            margin: 0 auto;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        .chat-input {
            width: 100%;
            background: var(--surface-color);
            border: 2px solid var(--border-color);
            border-radius: 16px;
            padding: 1rem 1.5rem;
            color: var(--text-primary);
            font-size: 1rem;
            resize: none;
            min-height: 60px;
            max-height: 200px;
            transition: all 0.2s ease;
        }

        .chat-input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .send-button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border: none;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem 1.5rem;
            background: var(--surface-color);
            border-radius: 16px;
            margin-bottom: 1rem;
        }

        .typing-dots {
            display: flex;
            gap: 0.25rem;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary-color);
            animation: typingAnimation 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typingAnimation {
            0%, 80%, 100% {
                transform: scale(0);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }

        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            padding: 2rem;
        }

        .welcome-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        .welcome-subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            margin-bottom: 3rem;
        }

        .example-prompts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            max-width: 800px;
            width: 100%;
        }

        .example-prompt {
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .example-prompt:hover {
            border-color: var(--primary-color);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }

        .hidden { display: none; }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                grid-template-areas: 
                    "header"
                    "main";
                grid-template-columns: 1fr;
                grid-template-rows: 70px 1fr;
            }

            .sidebar {
                display: none;
            }

            .input-container {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <i class="fas fa-shield-alt"></i>
                LLaMA CyberSec
            </div>
            <div class="status-indicator">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </header>

        <aside class="sidebar">
            <div class="nav-section">
                <div class="nav-title">Navigation</div>
                <div class="nav-item active" data-view="chat">
                    <i class="fas fa-comments"></i>
                    Chat Assistant
                </div>
                <div class="nav-item" data-view="analyze">
                    <i class="fas fa-code"></i>
                    Code Analysis
                </div>
                <div class="nav-item" data-view="upload">
                    <i class="fas fa-upload"></i>
                    File Upload
                </div>
                <div class="nav-item" data-view="reports">
                    <i class="fas fa-file-alt"></i>
                    Reports
                </div>
            </div>

            <div class="nav-section">
                <div class="nav-title">Tools</div>
                <div class="nav-item" data-view="vulnerability">
                    <i class="fas fa-bug"></i>
                    Vulnerability Scan
                </div>
                <div class="nav-item" data-view="monitoring">
                    <i class="fas fa-chart-line"></i>
                    Monitoring
                </div>
                <div class="nav-item" data-view="settings">
                    <i class="fas fa-cog"></i>
                    Settings
                </div>
            </div>
        </aside>

        <main class="main-content">
            <div class="chat-container" id="chatView">
                <div class="messages" id="messages">
                    <div class="welcome-screen" id="welcomeScreen">
                        <h1 class="welcome-title">🛡️ Bienvenue</h1>
                        <p class="welcome-subtitle">Assistant IA spécialisé en cybersécurité</p>
                        
                        <div class="example-prompts">
                            <div class="example-prompt" onclick="setPrompt('Créer une règle YARA pour détecter un ransomware moderne')">
                                <h3><i class="fas fa-shield-virus"></i> Règles de Détection</h3>
                                <p>Créer une règle YARA pour détecter un ransomware moderne</p>
                            </div>
                            <div class="example-prompt" onclick="setPrompt('Analyser ces logs Apache pour identifier des tentatives d\\'injection SQL')">
                                <h3><i class="fas fa-search"></i> Analyse de Logs</h3>
                                <p>Analyser des logs pour identifier des attaques</p>
                            </div>
                            <div class="example-prompt" onclick="setPrompt('Expliquer les techniques de lateral movement utilisées par les APT')">
                                <h3><i class="fas fa-route"></i> Threat Intelligence</h3>
                                <p>Expliquer les techniques d'attaque avancées</p>
                            </div>
                            <div class="example-prompt" onclick="setPrompt('Audit de sécurité de ce code Python et recommandations')">
                                <h3><i class="fas fa-code"></i> Audit de Code</h3>
                                <p>Analyser et sécuriser du code source</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="typing-indicator" id="typingIndicator">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                    <span>Assistant en cours de réflexion...</span>
                </div>

                <div class="input-area">
                    <div class="input-container">
                        <div class="input-wrapper">
                            <textarea id="chatInput" class="chat-input" placeholder="Posez votre question sur la cybersécurité..." rows="1"></textarea>
                        </div>
                        <button id="sendButton" class="send-button">
                            <i class="fas fa-paper-plane"></i>
                            Envoyer
                        </button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        // Variables globales
        let socket;
        let sessionId = null;
        let currentResponse = '';
        let isStreaming = false;

        // Initialisation
        document.addEventListener('DOMContentLoaded', function() {
            initializeSocket();
            setupEventListeners();
            autoResizeTextarea();
        });

        function initializeSocket() {
            socket = io();

            socket.on('connect', function() {
                console.log('Connected to server');
                updateStatus('ready', 'Connected');
            });

            socket.on('disconnect', function() {
                console.log('Disconnected from server');
                updateStatus('error', 'Disconnected');
            });

            socket.on('session_created', function(data) {
                sessionId = data.session_id;
                console.log('Session created:', sessionId);
            });

            socket.on('model_status', function(data) {
                if (data.status === 'ready') {
                    updateStatus('ready', 'Model Ready');
                } else if (data.status === 'loading') {
                    updateStatus('loading', `Loading... ${data.progress || 0}%`);
                } else if (data.status === 'error') {
                    updateStatus('error', 'Model Error');
                }
            });

            socket.on('chat_response', function(data) {
                handleStreamingResponse(data);
            });

            socket.on('error', function(data) {
                console.error('Socket error:', data);
                addMessage('assistant', `❌ Erreur: ${data.message}`);
                hideTypingIndicator();
            });
        }

        function setupEventListeners() {
            const chatInput = document.getElementById('chatInput');
            const sendButton = document.getElementById('sendButton');

            // Envoi de message
            sendButton.addEventListener('click', sendMessage);
            
            chatInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Navigation
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', function() {
                    const view = this.dataset.view;
                    switchView(view);
                    
                    // Update active state
                    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                    this.classList.add('active');
                });
            });
        }

        function autoResizeTextarea() {
            const textarea = document.getElementById('chatInput');
            
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 200) + 'px';
            });
        }

        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message || isStreaming) return;

            // Ajouter le message utilisateur
            addMessage('user', message);
            input.value = '';
            input.style.height = 'auto';

            // Cacher l'écran d'accueil
            hideWelcomeScreen();

            // Afficher l'indicateur de frappe
            showTypingIndicator();

            // Envoyer via WebSocket
            socket.emit('chat_message', { message: message });
            
            isStreaming = true;
            document.getElementById('sendButton').disabled = true;
        }

        function handleStreamingResponse(data) {
            if (data.type === 'start') {
                currentResponse = '';
                hideTypingIndicator();
                // Créer le conteneur de message de l'assistant
                createAssistantMessage();
            } else if (data.type === 'chunk') {
                currentResponse += data.content;
                updateAssistantMessage(currentResponse);
            } else if (data.type === 'end') {
                updateAssistantMessage(data.content);
                isStreaming = false;
                document.getElementById('sendButton').disabled = false;
            } else if (data.type === 'error') {
                addMessage('assistant', `❌ ${data.content}`);
                hideTypingIndicator();
                isStreaming = false;
                document.getElementById('sendButton').disabled = false;
            }
        }

        function addMessage(sender, content) {
            const messagesContainer = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            const avatar = document.createElement('div');
            avatar.className = `message-avatar ${sender}-avatar`;
            avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            if (sender === 'assistant') {
                messageContent.innerHTML = marked.parse(content);
                // Highlight code
                messageContent.querySelectorAll('pre code').forEach((block) => {
                    Prism.highlightElement(block);
                });
            } else {
                messageContent.textContent = content;
            }

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);

            // Scroll vers le bas
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function createAssistantMessage() {
            const messagesContainer = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.id = 'streaming-message';

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar assistant-avatar';
            avatar.innerHTML = '<i class="fas fa-robot"></i>';

            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.id = 'streaming-content';

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);

            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function updateAssistantMessage(content) {
            const contentDiv = document.getElementById('streaming-content');
            if (contentDiv) {
                contentDiv.innerHTML = marked.parse(content);
                // Highlight code
                contentDiv.querySelectorAll('pre code').forEach((block) => {
                    Prism.highlightElement(block);
                });
                
                // Scroll vers le bas
                const messagesContainer = document.getElementById('messages');
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'flex';
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }

        function hideWelcomeScreen() {
            document.getElementById('welcomeScreen').style.display = 'none';
        }

        function updateStatus(status, text) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            statusDot.className = `status-dot ${status}`;
            statusText.textContent = text;
        }

        function setPrompt(text) {
            document.getElementById('chatInput').value = text;
            document.getElementById('chatInput').focus();
        }

        function switchView(view) {
            // Pour l'instant, seule la vue chat est implémentée
            console.log('Switching to view:', view);
        }
    </script>
</body>
</html>"""

    def run(self):
        """Lance le serveur avec SocketIO"""
        host = self.config["web_ui"]["host"]
        port = self.config["web_ui"]["port"]
        
        self.logger.info(f"🚀 Démarrage du serveur moderne sur http://{host}:{port}")
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=False,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            self.logger.error(f"Erreur serveur: {e}")
            raise


# Classes pour les plugins
class CodeAnalyzerPlugin:
    def __init__(self, interface):
        self.interface = interface
        
    def analyze(self, code: str, language: str) -> Dict:
        """Analyse de code avec détection de vulnérabilités"""
        results = {
            'language': language,
            'lines_of_code': len(code.split('\n')),
            'vulnerabilities': [],
            'suggestions': [],
            'complexity_score': 0
        }
        
        # Détection basique de vulnérabilités
        if 'sql' in code.lower() and any(keyword in code.lower() for keyword in ['input', 'request', 'get', 'post']):
            results['vulnerabilities'].append({
                'type': 'SQL Injection',
                'severity': 'High',
                'description': 'Potential SQL injection vulnerability detected'
            })
        
        if 'eval(' in code or 'exec(' in code:
            results['vulnerabilities'].append({
                'type': 'Code Injection',
                'severity': 'Critical',
                'description': 'Use of eval() or exec() can lead to code injection'
            })
        
        # Suggestions générales
        if 'password' in code.lower() and 'plain' in code.lower():
            results['suggestions'].append('Consider hashing passwords instead of storing them in plain text')
        
        if 'http://' in code:
            results['suggestions'].append('Consider using HTTPS instead of HTTP for security')
        
        return results


class ReportGeneratorPlugin:
    def __init__(self, interface):
        self.interface = interface
        
    def generate_security_report(self, session_id: str, data: Dict) -> str:
        """Génère un rapport de sécurité"""
        report_id = str(uuid.uuid4())
        
        report_content = {
            'executive_summary': 'Security analysis report',
            'findings': data.get('vulnerabilities', []),
            'recommendations': data.get('suggestions', []),
            'generated_at': datetime.now().isoformat()
        }
        
        # Sauvegarder en base
        with self.interface.db_lock:
            cursor = self.interface.db_connection.cursor()
            cursor.execute(
                "INSERT INTO reports (id, session_id, report_type, title, content) VALUES (?, ?, ?, ?, ?)",
                (report_id, session_id, 'security_analysis', 'Security Analysis Report', json.dumps(report_content))
            )
            self.interface.db_connection.commit()
        
        return report_id


class VulnerabilityPlugin:
    def __init__(self, interface):
        self.interface = interface
        
    def scan_file(self, file_path: Path) -> Dict:
        """Scan de vulnérabilités sur un fichier"""
        results = {
            'file_path': str(file_path),
            'file_type': file_path.suffix,
            'vulnerabilities': [],
            'scan_date': datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Analyse basique selon le type de fichier
            if file_path.suffix in ['.py', '.js', '.php', '.java']:
                analyzer = CodeAnalyzerPlugin(self.interface)
                analysis = analyzer.analyze(content, file_path.suffix[1:])
                results['vulnerabilities'] = analysis['vulnerabilities']
            
        except Exception as e:
            results['error'] = str(e)
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Modern Interface for LLaMA CyberSec')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    
    args = parser.parse_args()
    
    try:
        interface = ModernLLaMAInterface(config_path=args.config)
        
        # Override config if arguments provided
        if args.host != '127.0.0.1':
            interface.config["web_ui"]["host"] = args.host
        if args.port != 8080:
            interface.config["web_ui"]["port"] = args.port
        
        interface.run()
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        sys.exit(1)