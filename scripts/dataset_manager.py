#!/usr/bin/env python3
"""
Gestionnaire de Dataset pour Fine-tuning LLaMA-3-8B Cybersécurité
Fonctionnalités:
- Création et gestion de datasets JSONL
- Parsing de sources locales et distantes
- Enrichissement automatique depuis Internet
- Validation et nettoyage des données
"""

import json
import os
import re
import requests
import git
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
from datetime import datetime
import hashlib
import random
from urllib.parse import urlparse
import tempfile
import subprocess
from bs4 import BeautifulSoup
import time

class DatasetManager:
    def __init__(self, dataset_path: str = "/app/datasets/initial_dataset.jsonl"):
        self.dataset_path = Path(dataset_path)
        self.sources_dir = Path("/app/datasets/sources")
        self.enriched_path = Path("/app/datasets/enriched_dataset.jsonl")
        
        # Créer les répertoires si nécessaires
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Sources publiques pour l'enrichissement automatique
        self.public_sources = {
            'sigma_rules': {
                'url': 'https://github.com/SigmaHQ/sigma.git',
                'type': 'git',
                'description': 'Règles Sigma officielles'
            },
            'yara_rules': {
                'url': 'https://github.com/Yara-Rules/rules.git',
                'type': 'git',
                'description': 'Collection de règles YARA'
            },
            'atomic_red_team': {
                'url': 'https://github.com/redcanaryco/atomic-red-team.git',
                'type': 'git',
                'description': 'Tests Atomic Red Team'
            },
            'mitre_attack': {
                'url': 'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
                'type': 'file',
                'description': 'MITRE ATT&CK Framework'
            },
            'cybersec_scripts': {
                'url': 'https://github.com/danielmiessler/SecLists.git',
                'type': 'git',
                'description': 'Listes de sécurité et scripts'
            },
            'owasp_guides': {
                'url': 'https://github.com/OWASP/CheatSheetSeries.git',
                'type': 'git',
                'description': 'Guides OWASP'
            }
        }
        
        # Patterns pour différents types de contenu
        self.content_patterns = {
            'sigma_rule': r'title:\s*(.+?)(?:\n|$)',
            'yara_rule': r'rule\s+(\w+)\s*{',
            'snort_rule': r'alert\s+\w+\s+.*?msg:"([^"]+)"',
            'python_function': r'def\s+(\w+)\s*\(',
            'bash_function': r'function\s+(\w+)\s*\(',
            'log_entry': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'hash_md5': r'\b[a-f0-9]{32}\b',
            'hash_sha256': r'\b[a-f0-9]{64}\b'
        }
    
    def setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/app/dataset_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_initial_dataset(self):
        """Crée le dataset initial s'il n'existe pas déjà"""
        if self.dataset_path.exists():
            self.logger.info(f"Dataset initial existe déjà: {self.dataset_path}")
            return
        
        self.logger.info("Création du dataset initial...")
        
        # Le dataset initial est déjà créé dans initial_dataset.jsonl
        # On va juste le copier s'il n'existe pas
        initial_examples = [
            {
                "instruction": "Créer une règle Sigma pour détecter les connexions RDP suspectes",
                "output": "Voici une règle Sigma pour détecter les connexions RDP suspectes :\n\ntitle: Suspicious RDP Connection\nstatus: experimental\nlogsource:\n    product: windows\n    service: security\ndetection:\n    selection:\n        EventID: 4624\n        LogonType: 10\n        WorkstationName: ''\n    timeframe: 5m\n    condition: selection | count() > 5\n\nCette règle détecte plus de 5 connexions RDP (LogonType 10) sans nom de workstation en 5 minutes, indicateur potentiel d'attaque par brute force."
            },
            {
                "instruction": "Analyser ce log Windows Event pour identifier une activité malveillante",
                "output": "Analyse du log Windows Event :\n\nEventID 4688 - Création de processus :\n- Processus: powershell.exe\n- Ligne de commande: powershell -enc JABhAD0AJwBoAHQAdABwADoALwAvAG\n- Utilisateur: SYSTEM\n- Processus parent: winlogon.exe\n\n**Indicateurs de compromission** :\n1. **Commande encodée** : Le paramètre -enc indique une commande PowerShell encodée en Base64\n2. **Exécution via SYSTEM** : Privilèges élevés suspects\n3. **Processus parent inhabituel** : winlogon.exe ne lance généralement pas PowerShell\n\n**Recommandations** :\n- Décoder la commande Base64 pour analyse\n- Vérifier les processus enfants créés\n- Examiner les connexions réseau établies\n- Corréler avec d'autres événements de sécurité"
            }
        ]
        
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            for example in initial_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Dataset initial créé avec {len(initial_examples)} exemples")
    
    def add_manual_example(self, instruction: str, output: str):
        """Ajoute un exemple manuellement au dataset"""
        example = {
            "instruction": instruction,
            "output": output
        }
        
        with open(self.dataset_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info("Exemple ajouté au dataset")
    
    def download_public_sources(self):
        """Télécharge les sources publiques"""
        self.logger.info("Téléchargement des sources publiques...")
        
        for source_name, source_info in self.public_sources.items():
            source_path = self.sources_dir / source_name
            
            try:
                if source_info['type'] == 'git':
                    if source_path.exists():
                        self.logger.info(f"Mise à jour du repo {source_name}...")
                        repo = git.Repo(source_path)
                        repo.remotes.origin.pull()
                    else:
                        self.logger.info(f"Clonage du repo {source_name}...")
                        git.Repo.clone_from(source_info['url'], source_path)
                
                elif source_info['type'] == 'file':
                    self.logger.info(f"Téléchargement du fichier {source_name}...")
                    response = requests.get(source_info['url'], timeout=30)
                    response.raise_for_status()
                    
                    filename = source_path / f"{source_name}.json"
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                
                self.logger.info(f"Source {source_name} téléchargée avec succès")
                time.sleep(1)  # Être respectueux avec les serveurs
                
            except Exception as e:
                self.logger.error(f"Erreur lors du téléchargement de {source_name}: {e}")
    
    def parse_sigma_rules(self, source_path: Path) -> List[Dict[str, str]]:
        """Parse les règles Sigma"""
        examples = []
        
        for sigma_file in source_path.rglob("*.yml"):
            try:
                with open(sigma_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extraction du titre et de la description
                title_match = re.search(r'title:\s*(.+?)(?:\n|$)', content)
                description_match = re.search(r'description:\s*(.+?)(?:\n|$)', content)
                
                if title_match:
                    title = title_match.group(1).strip()
                    description = description_match.group(1).strip() if description_match else ""
                    
                    instruction = f"Analyser et expliquer cette règle Sigma : {title}"
                    output = f"Analyse de la règle Sigma '{title}' :\n\n"
                    output += f"**Description** : {description}\n\n"
                    output += f"**Contenu de la règle** :\n```yaml\n{content[:500]}...\n```\n\n"
                    output += self.generate_sigma_explanation(content)
                    
                    examples.append({
                        "instruction": instruction,
                        "output": output
                    })
                    
                    if len(examples) >= 50:  # Limite pour éviter un dataset trop volumineux
                        break
                        
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing de {sigma_file}: {e}")
        
        return examples
    
    def generate_sigma_explanation(self, rule_content: str) -> str:
        """Génère une explication pour une règle Sigma"""
        explanation = "**Explication technique** :\n"
        
        # Analyse des éléments de la règle
        if 'EventID:' in rule_content:
            event_ids = re.findall(r'EventID:\s*(\d+)', rule_content)
            explanation += f"- Surveille les événements Windows : {', '.join(event_ids)}\n"
        
        if 'process' in rule_content.lower():
            explanation += "- Analyse les processus et leurs comportements\n"
        
        if 'network' in rule_content.lower():
            explanation += "- Surveille l'activité réseau\n"
        
        if 'registry' in rule_content.lower():
            explanation += "- Monitore les modifications du registre\n"
        
        if 'file' in rule_content.lower():
            explanation += "- Surveille les opérations sur fichiers\n"
        
        explanation += "\n**Utilisation** : Cette règle permet de détecter des activités suspectes "
        explanation += "correspondant aux techniques d'attaque connues et peut être intégrée "
        explanation += "dans des solutions SIEM comme Splunk, Elasticsearch ou Microsoft Sentinel."
        
        return explanation
    
    def parse_yara_rules(self, source_path: Path) -> List[Dict[str, str]]:
        """Parse les règles YARA"""
        examples = []
        
        for yara_file in source_path.rglob("*.yar"):
            try:
                with open(yara_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extraction des règles individuelles
                rule_matches = re.finditer(r'rule\s+(\w+)\s*{(.*?)}', content, re.DOTALL)
                
                for match in rule_matches:
                    rule_name = match.group(1)
                    rule_body = match.group(2)
                    
                    instruction = f"Expliquer cette règle YARA : {rule_name}"
                    output = f"Analyse de la règle YARA '{rule_name}' :\n\n"
                    output += f"**Nom** : {rule_name}\n\n"
                    output += f"**Contenu** :\n```\nrule {rule_name} {{\n{rule_body}\n}}\n```\n\n"
                    output += self.generate_yara_explanation(rule_body)
                    
                    examples.append({
                        "instruction": instruction,
                        "output": output
                    })
                    
                    if len(examples) >= 30:
                        break
                        
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing de {yara_file}: {e}")
        
        return examples
    
    def generate_yara_explanation(self, rule_body: str) -> str:
        """Génère une explication pour une règle YARA"""
        explanation = "**Explication** :\n"
        
        # Analyse des métadonnées
        if 'meta:' in rule_body:
            explanation += "- Contient des métadonnées descriptives\n"
        
        # Analyse des strings
        if 'strings:' in rule_body:
            explanation += "- Définit des chaînes de caractères à rechercher\n"
            
            # Compter les types de strings
            hex_strings = len(re.findall(r'\$\w+\s*=\s*{[0-9a-fA-F\s\?]+}', rule_body))
            text_strings = len(re.findall(r'\$\w+\s*=\s*"[^"]+"', rule_body))
            regex_strings = len(re.findall(r'\$\w+\s*=\s*/[^/]+/', rule_body))
            
            if hex_strings > 0:
                explanation += f"  - {hex_strings} pattern(s) hexadécimal(aux)\n"
            if text_strings > 0:
                explanation += f"  - {text_strings} chaîne(s) de texte\n"
            if regex_strings > 0:
                explanation += f"  - {regex_strings} expression(s) régulière(s)\n"
        
        # Analyse des conditions
        if 'condition:' in rule_body:
            explanation += "- Définit les conditions de détection\n"
            
            if 'any of' in rule_body:
                explanation += "  - Détection si au moins un pattern est trouvé\n"
            if 'all of' in rule_body:
                explanation += "  - Détection si tous les patterns sont trouvés\n"
            if 'filesize' in rule_body:
                explanation += "  - Prend en compte la taille du fichier\n"
        
        explanation += "\n**Utilisation** : Cette règle YARA peut être utilisée pour scanner "
        explanation += "des fichiers, de la mémoire ou des processus à la recherche de "
        explanation += "patterns spécifiques indicateurs de malware ou d'activité malveillante."
        
        return explanation
    
    def parse_script_files(self, source_path: Path) -> List[Dict[str, str]]:
        """Parse les scripts Python/Bash pour générer des exemples"""
        examples = []
        
        # Scripts Python
        for py_file in source_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extraction des fonctions
                functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):\s*"""([^"]+)"""', content, re.DOTALL)
                
                for func_name, docstring in functions[:3]:  # Limite à 3 par fichier
                    instruction = f"Analyser et expliquer cette fonction Python de sécurité : {func_name}"
                    output = f"Analyse de la fonction '{func_name}' :\n\n"
                    output += f"**Description** : {docstring.strip()}\n\n"
                    
                    # Extraction du code de la fonction
                    func_pattern = rf'def\s+{re.escape(func_name)}\s*\([^)]*\):.*?(?=\ndef|\nclass|\Z)'
                    func_match = re.search(func_pattern, content, re.DOTALL)
                    
                    if func_match:
                        func_code = func_match.group(0)[:500]  # Limiter la taille
                        output += f"**Code** :\n```python\n{func_code}...\n```\n\n"
                    
                    output += self.generate_function_explanation(func_name, docstring)
                    
                    examples.append({
                        "instruction": instruction,
                        "output": output
                    })
                    
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing de {py_file}: {e}")
        
        # Scripts Bash
        for bash_file in source_path.rglob("*.sh"):
            try:
                with open(bash_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extraction des fonctions Bash
                functions = re.findall(r'function\s+(\w+)\s*\(\s*\)\s*{([^}]+)}', content)
                
                for func_name, func_body in functions[:2]:  # Limite à 2 par fichier
                    instruction = f"Expliquer ce script Bash de sécurité : {func_name}"
                    output = f"Analyse du script Bash '{func_name}' :\n\n"
                    output += f"**Fonction** :\n```bash\nfunction {func_name}() {{\n{func_body}\n}}\n```\n\n"
                    output += self.generate_bash_explanation(func_body)
                    
                    examples.append({
                        "instruction": instruction,
                        "output": output
                    })
                    
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing de {bash_file}: {e}")
        
        return examples
    
    def generate_function_explanation(self, func_name: str, docstring: str) -> str:
        """Génère une explication pour une fonction"""
        explanation = f"**Fonctionnalité** : {docstring.strip()}\n\n"
        
        # Analyse basée sur le nom de la fonction
        if 'scan' in func_name.lower():
            explanation += "**Type** : Fonction de scanning/analyse\n"
        elif 'detect' in func_name.lower():
            explanation += "**Type** : Fonction de détection\n"
        elif 'parse' in func_name.lower():
            explanation += "**Type** : Fonction de parsing/analyse\n"
        elif 'check' in func_name.lower():
            explanation += "**Type** : Fonction de vérification\n"
        
        explanation += "**Utilisation** : Cette fonction peut être intégrée dans des scripts "
        explanation += "d'analyse de sécurité, des outils de monitoring ou des systèmes "
        explanation += "de détection d'intrusion automatisés."
        
        return explanation
    
    def generate_bash_explanation(self, func_body: str) -> str:
        """Génère une explication pour un script Bash"""
        explanation = "**Fonctionnalité** :\n"
        
        # Analyse des commandes utilisées
        if 'netstat' in func_body:
            explanation += "- Analyse des connexions réseau avec netstat\n"
        if 'ps' in func_body:
            explanation += "- Surveillance des processus avec ps\n"
        if 'grep' in func_body:
            explanation += "- Filtrage et recherche de patterns avec grep\n"
        if 'awk' in func_body:
            explanation += "- Traitement de texte avec awk\n"
        if 'curl' in func_body:
            explanation += "- Requêtes HTTP/HTTPS avec curl\n"
        if 'find' in func_body:
            explanation += "- Recherche de fichiers avec find\n"
        
        explanation += "\n**Cas d'usage** : Ce script peut être utilisé pour :\n"
        explanation += "- Monitoring système en temps réel\n"
        explanation += "- Détection d'anomalies\n"
        explanation += "- Collecte d'informations de sécurité\n"
        explanation += "- Automatisation de tâches de sécurité\n"
        
        return explanation
    
    def parse_log_files(self, source_path: Path) -> List[Dict[str, str]]:
        """Parse les fichiers de logs pour créer des exemples d'analyse"""
        examples = []
        
        for log_file in source_path.rglob("*.log"):
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:100]  # Limite à 100 lignes par fichier
                
                # Grouper les lignes par type de log
                grouped_logs = self.group_log_lines(lines)
                
                for log_type, log_lines in grouped_logs.items():
                    if len(log_lines) >= 3:  # Au moins 3 lignes pour créer un exemple
                        sample_logs = log_lines[:5]  # Prendre 5 lignes max
                        
                        instruction = f"Analyser ces logs {log_type} pour identifier les menaces potentielles"
                        output = f"Analyse des logs {log_type} :\n\n"
                        output += "**Échantillon de logs** :\n```\n"
                        output += '\n'.join(sample_logs)
                        output += "\n```\n\n"
                        output += self.generate_log_analysis(sample_logs, log_type)
                        
                        examples.append({
                            "instruction": instruction,
                            "output": output
                        })
                        
                        if len(examples) >= 20:  # Limite globale
                            break
                            
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing de {log_file}: {e}")
        
        return examples
    
    def group_log_lines(self, lines: List[str]) -> Dict[str, List[str]]:
        """Groupe les lignes de log par type"""
        grouped = {
            'apache': [],
            'nginx': [],
            'syslog': [],
            'auth': [],
            'generic': []
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Classification basique des logs
            if re.search(r'\d+\.\d+\.\d+\.\d+.*"(GET|POST|PUT|DELETE)', line):
                grouped['apache'].append(line)
            elif 'nginx' in line.lower():
                grouped['nginx'].append(line)
            elif re.search(r'(failed password|authentication failure|invalid user)', line.lower()):
                grouped['auth'].append(line)
            elif re.search(r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', line):
                grouped['syslog'].append(line)
            else:
                grouped['generic'].append(line)
        
        # Retourner seulement les groupes non vides
        return {k: v for k, v in grouped.items() if v}
    
    def generate_log_analysis(self, log_lines: List[str], log_type: str) -> str:
        """Génère une analyse des logs"""
        analysis = f"**Type de logs** : {log_type}\n\n"
        analysis += "**Éléments d'analyse** :\n"
        
        # Extraction d'IPs
        ips = set()
        for line in log_lines:
            ip_matches = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', line)
            ips.update(ip_matches)
        
        if ips:
            analysis += f"- Adresses IP identifiées : {', '.join(list(ips)[:3])}{'...' if len(ips) > 3 else ''}\n"
        
        # Détection de patterns suspects
        suspicious_patterns = []
        for line in log_lines:
            if re.search(r'(failed|error|unauthorized|denied|blocked)', line.lower()):
                suspicious_patterns.append("Échecs d'authentification ou erreurs")
            if re.search(r'(sql|script|alert|eval)', line.lower()):
                suspicious_patterns.append("Tentatives d'injection potentielles")
            if re.search(r'(\.\./|%2e%2e|traversal)', line.lower()):
                suspicious_patterns.append("Tentatives de directory traversal")
        
        if suspicious_patterns:
            analysis += f"- Patterns suspects détectés : {', '.join(set(suspicious_patterns))}\n"
        
        analysis += "\n**Recommandations** :\n"
        analysis += "- Corréler avec d'autres sources de logs\n"
        analysis += "- Vérifier la réputation des IPs suspectes\n"
        analysis += "- Analyser les tendances temporelles\n"
        analysis += "- Implémenter des règles de détection automatisées\n"
        
        return analysis
    
    def enrich_from_all_sources(self):
        """Enrichit le dataset depuis toutes les sources disponibles"""
        self.logger.info("Début de l'enrichissement du dataset...")
        
        # Télécharger les sources publiques
        self.download_public_sources()
        
        all_examples = []
        
        # Parser chaque type de source
        for source_name in self.public_sources.keys():
            source_path = self.sources_dir / source_name
            
            if not source_path.exists():
                continue
                
            try:
                self.logger.info(f"Parsing de la source {source_name}...")
                
                if source_name == 'sigma_rules':
                    examples = self.parse_sigma_rules(source_path)
                elif source_name == 'yara_rules':
                    examples = self.parse_yara_rules(source_path)
                elif source_name in ['cybersec_scripts', 'atomic_red_team']:
                    examples = self.parse_script_files(source_path)
                else:
                    # Parsing générique pour les autres sources
                    examples = self.parse_generic_source(source_path)
                
                self.logger.info(f"Source {source_name}: {len(examples)} exemples extraits")
                all_examples.extend(examples)
                
            except Exception as e:
                self.logger.error(f"Erreur lors du parsing de {source_name}: {e}")
        
        # Parser les sources locales
        local_examples = self.parse_local_sources()
        all_examples.extend(local_examples)
        
        # Mélanger et limiter le nombre d'exemples
        random.shuffle(all_examples)
        all_examples = all_examples[:1000]  # Limite à 1000 exemples
        
        # Sauvegarder le dataset enrichi
        with open(self.enriched_path, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Dataset enrichi créé avec {len(all_examples)} exemples: {self.enriched_path}")
    
    def parse_generic_source(self, source_path: Path) -> List[Dict[str, str]]:
        """Parse générique pour les sources non spécialisées"""
        examples = []
        
        # Recherche de fichiers de documentation
        for doc_file in source_path.rglob("*.md"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extraction des sections avec des titres
                sections = re.split(r'\n#+\s+', content)
                
                for section in sections[1:3]:  # Limite à 2 sections par fichier
                    lines = section.split('\n')
                    title = lines[0].strip()
                    
                    if len(title) > 10 and 'security' in title.lower():
                        instruction = f"Expliquer ce concept de sécurité : {title}"
                        output = f"Explication de '{title}' :\n\n"
                        output += '\n'.join(lines[1:10])  # Première partie du contenu
                        
                        examples.append({
                            "instruction": instruction,
                            "output": output
                        })
                        
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing de {doc_file}: {e}")
        
        return examples
    
    def parse_local_sources(self) -> List[Dict[str, str]]:
        """Parse les sources locales dans le répertoire sources/"""
        examples = []
        
        # Parser tous les fichiers dans le répertoire sources
        for file_path in self.sources_dir.rglob("*"):
            if file_path.is_file():
                try:
                    # Déterminer le type de fichier et le parser approprié
                    if file_path.suffix in ['.yml', '.yaml']:
                        examples.extend(self.parse_yaml_file(file_path))
                    elif file_path.suffix == '.py':
                        examples.extend(self.parse_python_file(file_path))
                    elif file_path.suffix == '.sh':
                        examples.extend(self.parse_bash_file(file_path))
                    elif file_path.suffix == '.log':
                        examples.extend(self.parse_log_file(file_path))
                    elif file_path.suffix in ['.txt', '.md']:
                        examples.extend(self.parse_text_file(file_path))
                        
                except Exception as e:
                    self.logger.debug(f"Erreur lors du parsing de {file_path}: {e}")
        
        return examples
    
    def parse_yaml_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse un fichier YAML (règle Sigma, etc.)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title_match = re.search(r'title:\s*(.+?)(?:\n|$)', content)
            if title_match:
                title = title_match.group(1).strip()
                
                instruction = f"Analyser cette règle de détection : {title}"
                output = f"Analyse de la règle '{title}' :\n\n"
                output += f"**Contenu** :\n```yaml\n{content}\n```\n\n"
                output += "Cette règle de détection permet d'identifier des activités suspectes "
                output += "dans les logs système et peut être utilisée dans un SIEM."
                
                return [{
                    "instruction": instruction,
                    "output": output
                }]
        except Exception as e:
            self.logger.debug(f"Erreur YAML {file_path}: {e}")
        
        return []
    
    def parse_python_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse un fichier Python"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraire la première fonction avec docstring
            func_match = re.search(r'def\s+(\w+)\s*\([^)]*\):\s*"""([^"]+)"""', content, re.DOTALL)
            if func_match:
                func_name = func_match.group(1)
                docstring = func_match.group(2)
                
                instruction = f"Analyser cette fonction Python de sécurité : {func_name}"
                output = f"Analyse de la fonction '{func_name}' :\n\n"
                output += f"**Description** : {docstring.strip()}\n\n"
                output += f"**Code** :\n```python\n{content[:800]}...\n```\n\n"
                output += "Cette fonction peut être utilisée dans des scripts d'analyse "
                output += "de sécurité ou des outils de monitoring automatisés."
                
                return [{
                    "instruction": instruction,
                    "output": output
                }]
        except Exception as e:
            self.logger.debug(f"Erreur Python {file_path}: {e}")
        
        return []
    
    def parse_bash_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse un fichier Bash"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = file_path.stem
            
            instruction = f"Expliquer ce script Bash de sécurité : {filename}"
            output = f"Analyse du script Bash '{filename}' :\n\n"
            output += f"**Script** :\n```bash\n{content[:600]}...\n```\n\n"
            output += self.generate_bash_explanation(content)
            
            return [{
                "instruction": instruction,
                "output": output
            }]
        except Exception as e:
            self.logger.debug(f"Erreur Bash {file_path}: {e}")
        
        return []
    
    def parse_log_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse un fichier de log"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:20]  # Première 20 lignes
            
            filename = file_path.stem
            
            instruction = f"Analyser ces logs de sécurité : {filename}"
            output = f"Analyse des logs '{filename}' :\n\n"
            output += "**Échantillon** :\n```\n"
            output += ''.join(lines[:10])
            output += "\n```\n\n"
            output += self.generate_log_analysis(lines, filename)
            
            return [{
                "instruction": instruction,
                "output": output
            }]
        except Exception as e:
            self.logger.debug(f"Erreur Log {file_path}: {e}")
        
        return []
    
    def parse_text_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse un fichier texte générique"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chercher des patterns de sécurité
            if any(keyword in content.lower() for keyword in ['security', 'vulnerability', 'attack', 'threat', 'malware']):
                filename = file_path.stem
                
                instruction = f"Expliquer ce document de sécurité : {filename}"
                output = f"Analyse du document '{filename}' :\n\n"
                output += f"**Contenu** :\n{content[:800]}...\n\n"
                output += "Ce document contient des informations pertinentes pour la "
                output += "cybersécurité et peut être utilisé pour la formation ou "
                output += "la sensibilisation aux menaces."
                
                return [{
                    "instruction": instruction,
                    "output": output
                }]
        except Exception as e:
            self.logger.debug(f"Erreur Text {file_path}: {e}")
        
        return []
    
    def validate_dataset(self, dataset_path: str = None):
        """Valide la structure et le contenu du dataset"""
        if dataset_path is None:
            dataset_path = self.dataset_path
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            self.logger.error(f"Dataset non trouvé: {dataset_path}")
            return False
        
        valid_count = 0
        invalid_count = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Vérifier la structure
                    if not isinstance(data, dict):
                        self.logger.warning(f"Ligne {line_num}: Pas un objet JSON")
                        invalid_count += 1
                        continue
                    
                    if 'instruction' not in data or 'output' not in data:
                        self.logger.warning(f"Ligne {line_num}: Champs manquants")
                        invalid_count += 1
                        continue
                    
                    # Vérifier la longueur
                    if len(data['instruction']) < 10 or len(data['output']) < 20:
                        self.logger.warning(f"Ligne {line_num}: Contenu trop court")
                        invalid_count += 1
                        continue
                    
                    valid_count += 1
                    
                except json.JSONDecodeError:
                    self.logger.warning(f"Ligne {line_num}: JSON invalide")
                    invalid_count += 1
        
        total = valid_count + invalid_count
        self.logger.info(f"Validation terminée: {valid_count}/{total} exemples valides ({valid_count/total*100:.1f}%)")
        
        return valid_count > 0
    
    def list_dataset_entries(self, limit: int = 10):
        """Liste les entrées du dataset"""
        if not self.dataset_path.exists():
            self.logger.error("Dataset non trouvé")
            return
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    print(f"\n--- Exemple {i+1} ---")
                    print(f"Instruction: {data['instruction'][:100]}...")
                    print(f"Output: {data['output'][:100]}...")
                except json.JSONDecodeError:
                    print(f"Ligne {i+1}: JSON invalide")
    
    def get_dataset_stats(self):
        """Affiche les statistiques du dataset"""
        if not self.dataset_path.exists():
            self.logger.error("Dataset non trouvé")
            return
        
        total_examples = 0
        total_instruction_chars = 0
        total_output_chars = 0
        topics = {}
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    total_examples += 1
                    total_instruction_chars += len(data['instruction'])
                    total_output_chars += len(data['output'])
                    
                    # Analyse des sujets
                    instruction_lower = data['instruction'].lower()
                    for keyword in ['sigma', 'yara', 'python', 'bash', 'log', 'xss', 'sql', 'malware']:
                        if keyword in instruction_lower:
                            topics[keyword] = topics.get(keyword, 0) + 1
                            
                except json.JSONDecodeError:
                    continue
        
        print(f"\n=== Statistiques du Dataset ===")
        print(f"Nombre total d'exemples: {total_examples}")
        print(f"Longueur moyenne instruction: {total_instruction_chars//total_examples if total_examples > 0 else 0} caractères")
        print(f"Longueur moyenne output: {total_output_chars//total_examples if total_examples > 0 else 0} caractères")
        print(f"\nSujets couverts:")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic}: {count} exemples")

def main():
    parser = argparse.ArgumentParser(description='Gestionnaire de Dataset pour LLaMA-3-8B Cybersécurité')
    parser.add_argument('--create-initial', action='store_true', help='Créer le dataset initial')
    parser.add_argument('--add-example', nargs=2, metavar=('INSTRUCTION', 'OUTPUT'), help='Ajouter un exemple manuel')
    parser.add_argument('--enrich-all', action='store_true', help='Enrichir depuis toutes les sources')
    parser.add_argument('--validate', action='store_true', help='Valider le dataset')
    parser.add_argument('--list', type=int, default=10, help='Lister les entrées du dataset')
    parser.add_argument('--stats', action='store_true', help='Afficher les statistiques')
    parser.add_argument('--dataset-path', help='Chemin vers le dataset')
    
    args = parser.parse_args()
    
    # Initialiser le gestionnaire
    dataset_path = args.dataset_path or "/app/datasets/initial_dataset.jsonl"
    manager = DatasetManager(dataset_path)
    
    if args.create_initial:
        manager.create_initial_dataset()
    
    if args.add_example:
        instruction, output = args.add_example
        manager.add_manual_example(instruction, output)
    
    if args.enrich_all:
        manager.enrich_from_all_sources()
    
    if args.validate:
        manager.validate_dataset()
    
    if args.list:
        manager.list_dataset_entries(args.list)
    
    if args.stats:
        manager.get_dataset_stats()
    
    if not any([args.create_initial, args.add_example, args.enrich_all, args.validate, args.list, args.stats]):
        print("Utilisez --help pour voir les options disponibles")

if __name__ == "__main__":
    main()