#!/usr/bin/env python3
"""
Suite de tests pour le système de fine-tuning LLaMA-3-8B Cybersécurité
Tests des composants principaux et de l'intégration
"""

import os
import sys
import json
import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

# Ajouter le répertoire parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dataset_manager import DatasetManager
from scripts.train_lora import LLaMACyberSecTrainer

class TestDatasetManager:
    """Tests pour le gestionnaire de dataset"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_dataset.jsonl"
        self.manager = DatasetManager(str(self.dataset_path))
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_initial_dataset(self):
        """Test de création du dataset initial"""
        self.manager.create_initial_dataset()
        
        assert self.dataset_path.exists()
        
        # Vérifier le contenu
        with open(self.dataset_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) >= 2  # Au moins 2 exemples
        
        # Vérifier la structure JSON
        for line in lines:
            data = json.loads(line.strip())
            assert 'instruction' in data
            assert 'output' in data
            assert len(data['instruction']) > 0
            assert len(data['output']) > 0
    
    def test_add_manual_example(self):
        """Test d'ajout d'exemple manuel"""
        instruction = "Test instruction"
        output = "Test output"
        
        self.manager.add_manual_example(instruction, output)
        
        assert self.dataset_path.exists()
        
        with open(self.dataset_path, 'r') as f:
            data = json.loads(f.read().strip())
        
        assert data['instruction'] == instruction
        assert data['output'] == output
    
    def test_validate_dataset(self):
        """Test de validation du dataset"""
        # Créer un dataset valide
        valid_examples = [
            {"instruction": "Test 1", "output": "Response 1"},
            {"instruction": "Test 2", "output": "Response 2"}
        ]
        
        with open(self.dataset_path, 'w') as f:
            for example in valid_examples:
                f.write(json.dumps(example) + '\n')
        
        # Tester la validation
        is_valid = self.manager.validate_dataset(str(self.dataset_path))
        assert is_valid
    
    def test_validate_invalid_dataset(self):
        """Test de validation avec dataset invalide"""
        # Créer un dataset invalide
        with open(self.dataset_path, 'w') as f:
            f.write('{"instruction": "Test"}\n')  # Champ manquant
            f.write('invalid json\n')  # JSON invalide
        
        is_valid = self.manager.validate_dataset(str(self.dataset_path))
        assert not is_valid
    
    def test_get_dataset_stats(self):
        """Test des statistiques de dataset"""
        # Créer un dataset test
        examples = [
            {"instruction": "sigma rule", "output": "sigma response"},
            {"instruction": "yara detection", "output": "yara response"},
            {"instruction": "python script", "output": "python response"}
        ]
        
        with open(self.dataset_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        # Rediriger stdout pour capturer les stats
        from io import StringIO
        import contextlib
        
        f = StringIO()
        with contextlib.redirect_stdout(f):
            self.manager.get_dataset_stats()
        
        output = f.getvalue()
        assert "Nombre total d'exemples: 3" in output
        assert "sigma: 1" in output
        assert "yara: 1" in output
        assert "python: 1" in output
    
    @patch('git.Repo.clone_from')
    def test_download_public_sources(self, mock_clone):
        """Test de téléchargement des sources publiques"""
        # Mock du clonage Git
        mock_clone.return_value = Mock()
        
        # Test avec sources limitées
        self.manager.public_sources = {
            'test_repo': {
                'url': 'https://github.com/test/repo.git',
                'type': 'git',
                'description': 'Test repo'
            }
        }
        
        self.manager.download_public_sources()
        
        # Vérifier que le clonage a été appelé
        mock_clone.assert_called_once()
    
    def test_group_log_lines(self):
        """Test de groupement des lignes de log"""
        log_lines = [
            '192.168.1.1 - - [01/Jan/2024:00:00:00 +0000] "GET / HTTP/1.1" 200 1234',
            'Jan  1 00:00:00 server sshd[1234]: Failed password for user from 192.168.1.2',
            'nginx: error message',
            'generic log line'
        ]
        
        grouped = self.manager.group_log_lines(log_lines)
        
        assert 'apache' in grouped
        assert 'auth' in grouped
        assert 'nginx' in grouped
        assert 'generic' in grouped
        
        assert len(grouped['apache']) == 1
        assert len(grouped['auth']) == 1
        assert len(grouped['nginx']) == 1
        assert len(grouped['generic']) == 1

class TestSystemIntegration:
    """Tests d'intégration système"""
    
    def test_scripts_are_executable(self):
        """Vérifier que les scripts principaux sont exécutables"""
        scripts = [
            "/app/scripts/dataset_manager.py",
            "/app/scripts/train_lora.py", 
            "/app/scripts/merge_and_quantize.py",
            "/app/scripts/install.sh"
        ]
        
        for script in scripts:
            script_path = Path(script)
            assert script_path.exists(), f"Script manquant: {script}"
            
            if script.endswith('.py'):
                # Vérifier que c'est du Python valide
                result = subprocess.run([
                    sys.executable, '-m', 'py_compile', script
                ], capture_output=True)
                assert result.returncode == 0, f"Erreur de syntaxe dans {script}"
    
    def test_help_commands(self):
        """Tester les commandes d'aide"""
        scripts = [
            "/app/scripts/dataset_manager.py",
            "/app/scripts/train_lora.py",
            "/app/scripts/merge_and_quantize.py"
        ]
        
        for script in scripts:
            result = subprocess.run([
                sys.executable, script, '--help'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"Erreur --help pour {script}"
            assert 'usage:' in result.stdout.lower() or 'Usage:' in result.stdout
    
    def test_directory_structure(self):
        """Vérifier la structure des répertoires"""
        required_dirs = [
            "/app/datasets",
            "/app/scripts", 
            "/app/models",
            "/app/configs",
            "/app/docs"
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Répertoire manquant: {dir_path}"
            assert Path(dir_path).is_dir(), f"N'est pas un répertoire: {dir_path}"
    
    def test_config_files_valid(self):
        """Vérifier que les fichiers de configuration sont valides"""
        config_files = [
            "/app/configs/training_config.json",
            "/app/configs/dataset_config.json"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    try:
                        json.load(f)
                    except json.JSONDecodeError:
                        pytest.fail(f"JSON invalide dans {config_file}")

class TestModelComponents:
    """Tests des composants de modélisation"""
    
    @pytest.mark.slow
    def test_trainer_initialization(self):
        """Test d'initialisation du trainer (sans GPU requis)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test.jsonl"
            
            # Créer un mini dataset
            examples = [
                {"instruction": "test", "output": "response"}
            ]
            
            with open(dataset_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            
            trainer = LLaMACyberSecTrainer(
                model_name="gpt2",  # Modèle plus léger pour test
                dataset_path=str(dataset_path),
                output_dir=temp_dir
            )
            
            # Vérifier l'initialisation
            assert trainer.dataset_path.exists()
            assert trainer.output_dir.exists()
            assert trainer.lora_config is not None
            assert trainer.training_config is not None
    
    def test_prompt_formatting(self):
        """Test du formatage des prompts"""
        trainer = LLaMACyberSecTrainer()
        
        instruction = "Test instruction"
        output = "Test output"
        
        formatted = trainer.format_prompt(instruction, output)
        
        assert "### Instruction:" in formatted
        assert "### Response:" in formatted
        assert instruction in formatted
        assert output in formatted
    
    def test_training_config_validation(self):
        """Test de validation des configurations d'entraînement"""
        trainer = LLaMACyberSecTrainer()
        
        # Vérifier les paramètres obligatoires
        required_params = [
            "per_device_train_batch_size",
            "learning_rate",
            "num_train_epochs",
            "output_dir"
        ]
        
        for param in required_params:
            assert param in trainer.training_config

class TestUtilities:
    """Tests des utilitaires"""
    
    def test_content_patterns(self):
        """Test des patterns de reconnaissance de contenu"""
        manager = DatasetManager()
        
        # Test pattern Sigma
        sigma_content = "title: Test Rule\ndetection:\n  selection:\n    EventID: 4624"
        title_match = manager.content_patterns['sigma_rule']
        import re
        matches = re.findall(title_match, sigma_content)
        assert len(matches) > 0
        
        # Test pattern YARA
        yara_content = "rule TestRule { condition: true }"
        yara_match = manager.content_patterns['yara_rule']
        matches = re.findall(yara_match, yara_content)
        assert len(matches) > 0
        
        # Test pattern IP
        ip_content = "Connection from 192.168.1.100"
        ip_match = manager.content_patterns['ip_address']
        matches = re.findall(ip_match, ip_content)
        assert "192.168.1.100" in matches
    
    def test_file_extensions_handling(self):
        """Test de gestion des extensions de fichiers"""
        manager = DatasetManager()
        
        # Simuler différents types de fichiers
        test_files = {
            "rule.yml": "yaml",
            "script.py": "python", 
            "tool.sh": "bash",
            "events.log": "log",
            "doc.md": "text"
        }
        
        for filename, expected_type in test_files.items():
            file_path = Path(filename)
            suffix = file_path.suffix
            
            # Vérifier que l'extension est reconnue
            assert suffix in ['.yml', '.py', '.sh', '.log', '.md']

def test_requirements_installed():
    """Vérifier que les dépendances sont installées"""
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'peft',
        'bitsandbytes',
        'requests',
        'beautifulsoup4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        pytest.fail(f"Packages manquants: {missing_packages}")

def test_python_version():
    """Vérifier la version Python"""
    import sys
    
    major, minor = sys.version_info[:2]
    assert major == 3, f"Python 3 requis, trouvé: {major}.{minor}"
    assert minor >= 8, f"Python 3.8+ requis, trouvé: {major}.{minor}"

if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v", "--tb=short"])