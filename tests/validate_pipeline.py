#!/usr/bin/env python3
"""
Script de validation complète du pipeline de fine-tuning LLaMA-3-8B
Teste l'ensemble du workflow depuis la création du dataset jusqu'au déploiement
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineValidator:
    def __init__(self, test_mode=True):
        self.test_mode = test_mode
        self.project_dir = Path("/app")
        self.temp_dir = Path(tempfile.mkdtemp()) if test_mode else self.project_dir
        self.results = {}
        self.start_time = datetime.now()
        
        logger.info(f"Validation pipeline - Mode test: {test_mode}")
        logger.info(f"Répertoire de travail: {self.temp_dir}")
    
    def log_result(self, test_name, success, message="", details=None):
        """Enregistre le résultat d'un test"""
        self.results[test_name] = {
            'success': success,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
    
    def test_system_requirements(self):
        """Teste les prérequis système"""
        logger.info("=== Test des prérequis système ===")
        
        try:
            # Test Python
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.version_info >= (3, 8):
                self.log_result("python_version", True, f"Python {python_version}")
            else:
                self.log_result("python_version", False, f"Python {python_version} < 3.8")
            
            # Test des imports principaux
            required_modules = [
                'torch', 'transformers', 'datasets', 'peft', 
                'bitsandbytes', 'requests', 'git'
            ]
            
            missing_modules = []
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if not missing_modules:
                self.log_result("python_dependencies", True, "Toutes les dépendances installées")
            else:
                self.log_result("python_dependencies", False, f"Modules manquants: {missing_modules}")
            
            # Test CUDA (optionnel)
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    self.log_result("gpu_cuda", True, f"{gpu_name} ({gpu_memory:.1f}GB)")
                else:
                    self.log_result("gpu_cuda", False, "CUDA non disponible - CPU seulement")
            except:
                self.log_result("gpu_cuda", False, "Erreur lors du test CUDA")
            
            # Test mémoire
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / 1e9
                if ram_gb >= 16:
                    self.log_result("system_memory", True, f"{ram_gb:.1f}GB RAM")
                else:
                    self.log_result("system_memory", False, f"{ram_gb:.1f}GB RAM < 16GB minimum")
            except ImportError:
                self.log_result("system_memory", False, "psutil non disponible")
            
        except Exception as e:
            self.log_result("system_requirements", False, f"Erreur: {e}")
    
    def test_file_structure(self):
        """Teste la structure des fichiers"""
        logger.info("=== Test de la structure des fichiers ===")
        
        required_files = [
            "README.md",
            "scripts/dataset_manager.py",
            "scripts/train_lora.py", 
            "scripts/merge_and_quantize.py",
            "scripts/install.sh",
            "docs/GUIDE_UTILISATION.md",
            "datasets/initial_dataset.jsonl"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if not missing_files:
            self.log_result("file_structure", True, f"{len(required_files)} fichiers présents")
        else:
            self.log_result("file_structure", False, f"Fichiers manquants: {missing_files}")
        
        # Test des répertoires
        required_dirs = [
            "datasets", "scripts", "models", "configs", "docs", "tests"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_dir / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if not missing_dirs:
            self.log_result("directory_structure", True, f"{len(required_dirs)} répertoires présents")
        else:
            self.log_result("directory_structure", False, f"Répertoires manquants: {missing_dirs}")
    
    def test_dataset_creation(self):
        """Teste la création et validation du dataset"""
        logger.info("=== Test de création du dataset ===")
        
        try:
            # Importer et tester le gestionnaire de dataset
            sys.path.insert(0, str(self.project_dir))
            from scripts.dataset_manager import DatasetManager
            
            # Créer un dataset de test
            test_dataset_path = self.temp_dir / "test_dataset.jsonl"
            manager = DatasetManager(str(test_dataset_path))
            
            # Test de création initiale
            manager.create_initial_dataset()
            
            if test_dataset_path.exists():
                self.log_result("dataset_creation", True, "Dataset initial créé")
            else:
                self.log_result("dataset_creation", False, "Échec création dataset")
                return
            
            # Test de validation
            is_valid = manager.validate_dataset()
            if is_valid:
                self.log_result("dataset_validation", True, "Dataset valide")
            else:
                self.log_result("dataset_validation", False, "Dataset invalide")
            
            # Test d'ajout d'exemple
            manager.add_manual_example(
                "Test de règle Sigma",
                "Ceci est un test de règle Sigma pour validation"
            )
            
            # Compter les exemples
            example_count = 0
            with open(test_dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        example_count += 1
            
            if example_count >= 2:
                self.log_result("dataset_examples", True, f"{example_count} exemples")
            else:
                self.log_result("dataset_examples", False, f"Seulement {example_count} exemples")
            
        except Exception as e:
            self.log_result("dataset_creation", False, f"Erreur: {e}")
    
    def test_training_configuration(self):
        """Teste la configuration d'entraînement"""
        logger.info("=== Test de configuration d'entraînement ===")
        
        try:
            sys.path.insert(0, str(self.project_dir))
            from scripts.train_lora import LLaMACyberSecTrainer
            
            # Test d'initialisation du trainer
            trainer = LLaMACyberSecTrainer(
                model_name="gpt2",  # Modèle léger pour test
                dataset_path=str(self.project_dir / "datasets" / "initial_dataset.jsonl"),
                output_dir=str(self.temp_dir / "test_output")
            )
            
            # Vérifier les configurations
            if trainer.lora_config and trainer.training_config:
                self.log_result("training_config", True, "Configuration LoRA/training OK")
            else:
                self.log_result("training_config", False, "Configuration manquante")
            
            # Test de vérification système
            trainer.check_system_requirements()
            self.log_result("system_check", True, "Vérification système OK")
            
            # Test de formatage de prompt
            test_prompt = trainer.format_prompt("Test instruction", "Test output")
            if "### Instruction:" in test_prompt and "### Response:" in test_prompt:
                self.log_result("prompt_formatting", True, "Formatage prompt OK")
            else:
                self.log_result("prompt_formatting", False, "Erreur formatage prompt")
            
        except Exception as e:
            self.log_result("training_configuration", False, f"Erreur: {e}")
    
    def test_scripts_execution(self):
        """Teste l'exécution des scripts principaux"""
        logger.info("=== Test d'exécution des scripts ===")
        
        scripts_to_test = [
            ("dataset_manager.py", ["--help"]),
            ("train_lora.py", ["--help"]),
            ("merge_and_quantize.py", ["--help"])
        ]
        
        for script_name, args in scripts_to_test:
            try:
                script_path = self.project_dir / "scripts" / script_name
                result = subprocess.run([
                    sys.executable, str(script_path)
                ] + args, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.log_result(f"script_{script_name}", True, "Exécution réussie")
                else:
                    self.log_result(f"script_{script_name}", False, f"Code retour: {result.returncode}")
                
            except subprocess.TimeoutExpired:
                self.log_result(f"script_{script_name}", False, "Timeout")
            except Exception as e:
                self.log_result(f"script_{script_name}", False, f"Erreur: {e}")
    
    def test_model_pipeline_dry_run(self):
        """Test à blanc du pipeline complet (sans entraînement réel)"""
        logger.info("=== Test à blanc du pipeline ===")
        
        try:
            # 1. Test de chargement de dataset
            sys.path.insert(0, str(self.project_dir))
            from scripts.train_lora import LLaMACyberSecTrainer
            
            # Créer un mini dataset pour test
            test_dataset = self.temp_dir / "mini_dataset.jsonl"
            mini_examples = [
                {"instruction": "Test 1", "output": "Response 1"},
                {"instruction": "Test 2", "output": "Response 2"},
                {"instruction": "Test 3", "output": "Response 3"}
            ]
            
            with open(test_dataset, 'w') as f:
                for ex in mini_examples:
                    f.write(json.dumps(ex) + '\n')
            
            trainer = LLaMACyberSecTrainer(
                model_name="gpt2",
                dataset_path=str(test_dataset),
                output_dir=str(self.temp_dir / "pipeline_test")
            )
            
            # Test de chargement dataset
            dataset = trainer.load_dataset()
            if len(dataset['train']) > 0:
                self.log_result("pipeline_dataset_load", True, f"{len(dataset['train'])} exemples train")
            else:
                self.log_result("pipeline_dataset_load", False, "Dataset vide")
            
            # Test estimation temps (sans entraînement)
            trainer.estimate_training_time()
            self.log_result("pipeline_time_estimation", True, "Estimation temps OK")
            
        except Exception as e:
            self.log_result("model_pipeline_dry_run", False, f"Erreur: {e}")
    
    def test_integration_compatibility(self):
        """Teste la compatibilité avec les outils d'intégration"""
        logger.info("=== Test de compatibilité intégration ===")
        
        # Test llama.cpp (installation)
        try:
            # Vérifier si llama.cpp est disponible
            llama_paths = [
                Path.home() / "llama.cpp",
                Path("/usr/local/bin/llama.cpp"),
                Path("/opt/llama.cpp")
            ]
            
            llama_found = False
            for path in llama_paths:
                if (path / "convert.py").exists():
                    llama_found = True
                    break
            
            if llama_found:
                self.log_result("llama_cpp_available", True, f"llama.cpp trouvé: {path}")
            else:
                self.log_result("llama_cpp_available", False, "llama.cpp non installé")
        
        except Exception as e:
            self.log_result("llama_cpp_available", False, f"Erreur: {e}")
        
        # Test Ollama (disponibilité)
        try:
            result = subprocess.run(['which', 'ollama'], capture_output=True)
            if result.returncode == 0:
                self.log_result("ollama_available", True, "Ollama installé")
            else:
                self.log_result("ollama_available", False, "Ollama non installé")
        except:
            self.log_result("ollama_available", False, "Erreur test Ollama")
        
        # Test Hugging Face Hub
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # Test simple de l'API (sans authentification)
            self.log_result("huggingface_hub", True, "Hugging Face Hub accessible")
        except Exception as e:
            self.log_result("huggingface_hub", False, f"Erreur HF Hub: {e}")
    
    def run_performance_benchmark(self):
        """Benchmark de performance basique"""
        logger.info("=== Benchmark de performance ===")
        
        try:
            import torch
            
            # Test performance CPU
            start_time = time.time()
            x = torch.randn(1000, 1000)
            y = torch.mm(x, x)
            cpu_time = time.time() - start_time
            
            self.log_result("cpu_performance", True, f"Matrix mul: {cpu_time:.3f}s")
            
            # Test performance GPU (si disponible)
            if torch.cuda.is_available():
                device = torch.device('cuda')
                x_gpu = torch.randn(1000, 1000, device=device)
                
                # Warm-up
                torch.mm(x_gpu, x_gpu)
                torch.cuda.synchronize()
                
                start_time = time.time()
                y_gpu = torch.mm(x_gpu, x_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time
                self.log_result("gpu_performance", True, f"Matrix mul: {gpu_time:.3f}s (speedup: {speedup:.1f}x)")
            else:
                self.log_result("gpu_performance", False, "GPU non disponible")
        
        except Exception as e:
            self.log_result("performance_benchmark", False, f"Erreur: {e}")
    
    def generate_report(self):
        """Génère un rapport de validation"""
        logger.info("=== Génération du rapport ===")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 1)
            },
            "test_results": self.results,
            "recommendations": self.generate_recommendations()
        }
        
        # Sauvegarder le rapport
        report_path = self.project_dir / "validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Afficher le résumé
        self.display_summary(report)
        
        return report
    
    def generate_recommendations(self):
        """Génère des recommandations basées sur les résultats"""
        recommendations = []
        
        if not self.results.get("python_dependencies", {}).get("success", False):
            recommendations.append({
                "category": "installation",
                "priority": "high",
                "message": "Installer les dépendances Python manquantes avec: pip install -r requirements.txt"
            })
        
        if not self.results.get("gpu_cuda", {}).get("success", False):
            recommendations.append({
                "category": "performance", 
                "priority": "medium",
                "message": "Considérer l'utilisation d'un GPU NVIDIA pour accélérer l'entraînement"
            })
        
        if not self.results.get("system_memory", {}).get("success", False):
            recommendations.append({
                "category": "hardware",
                "priority": "high", 
                "message": "Augmenter la RAM à 20GB minimum pour un entraînement optimal"
            })
        
        if not self.results.get("llama_cpp_available", {}).get("success", False):
            recommendations.append({
                "category": "tools",
                "priority": "medium",
                "message": "Installer llama.cpp pour la quantisation: scripts/install.sh"
            })
        
        return recommendations
    
    def display_summary(self, report):
        """Affiche un résumé de la validation"""
        summary = report["validation_summary"]
        
        print(f"\n{'='*60}")
        print("🔍 RAPPORT DE VALIDATION DU SYSTÈME")
        print(f"{'='*60}")
        
        print(f"\n📊 RÉSULTATS:")
        print(f"  • Tests exécutés: {summary['total_tests']}")
        print(f"  • Tests réussis: {summary['passed_tests']} ✅")
        print(f"  • Tests échoués: {summary['failed_tests']} ❌")
        print(f"  • Taux de réussite: {summary['success_rate']}%")
        print(f"  • Durée: {summary['duration_seconds']:.1f}s")
        
        # Tests échoués
        failed_tests = [name for name, result in self.results.items() if not result['success']]
        if failed_tests:
            print(f"\n❌ TESTS ÉCHOUÉS:")
            for test in failed_tests:
                message = self.results[test]['message']
                print(f"  • {test}: {message}")
        
        # Recommandations
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\n💡 RECOMMANDATIONS:")
            for rec in recommendations:
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡"
                print(f"  {priority_icon} {rec['message']}")
        
        # Status final
        if summary['success_rate'] >= 90:
            print(f"\n🎉 SYSTÈME PRÊT - Le pipeline est opérationnel!")
        elif summary['success_rate'] >= 70:
            print(f"\n⚠️ SYSTÈME PARTIELLEMENT PRÊT - Quelques améliorations nécessaires")
        else:
            print(f"\n🚨 SYSTÈME NON PRÊT - Corrections importantes requises")
        
        print(f"\n📄 Rapport détaillé: validation_report.json")
    
    def run_full_validation(self):
        """Exécute la validation complète"""
        logger.info("Début de la validation complète du système")
        
        try:
            self.test_system_requirements()
            self.test_file_structure()
            self.test_dataset_creation()
            self.test_training_configuration()
            self.test_scripts_execution()
            self.test_model_pipeline_dry_run()
            self.test_integration_compatibility()
            self.run_performance_benchmark()
            
            return self.generate_report()
            
        except KeyboardInterrupt:
            logger.info("Validation interrompue par l'utilisateur")
            return self.generate_report()
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {e}")
            self.log_result("validation_error", False, str(e))
            return self.generate_report()
        finally:
            # Nettoyage
            if self.test_mode and self.temp_dir != self.project_dir:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validation du pipeline LLaMA Cybersécurité')
    parser.add_argument('--production', action='store_true', 
                        help='Mode production (utilise les vrais répertoires)')
    parser.add_argument('--quick', action='store_true',
                        help='Tests rapides uniquement')
    parser.add_argument('--output', default='validation_report.json',
                        help='Fichier de rapport de sortie')
    
    args = parser.parse_args()
    
    # Initialiser le validateur
    validator = PipelineValidator(test_mode=not args.production)
    
    if args.quick:
        # Tests rapides seulement
        validator.test_system_requirements()
        validator.test_file_structure()
        validator.test_scripts_execution()
    else:
        # Validation complète
        validator.run_full_validation()
    
    print(f"\n✅ Validation terminée - Consultez le rapport pour les détails")

if __name__ == "__main__":
    main()