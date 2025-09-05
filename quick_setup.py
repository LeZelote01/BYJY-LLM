#!/usr/bin/env python3
"""
Configuration rapide pour LLaMA-3-8B Cybersécurité
Script interactif pour guider l'installation selon le besoin utilisateur
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class QuickSetup:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.config = {}
        
    def welcome(self):
        """Affiche le message d'accueil"""
        print("""
🛡️  LLaMA-3-8B CYBERSÉCURITÉ - CONFIGURATION RAPIDE
═══════════════════════════════════════════════════

Bienvenue ! Ce script va vous guider pour configurer le système
selon vos besoins spécifiques.

Options disponibles :
1. 📊 Configuration complète (recommandée)
2. ☁️  Entraînement cloud uniquement  
3. 🏠 Installation locale uniquement
4. 🔧 Configuration avancée personnalisée
        """)
    
    def get_user_choice(self) -> str:
        """Obtient le choix de l'utilisateur"""
        while True:
            choice = input("\nVotre choix (1-4) : ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            print("❌ Choix invalide. Veuillez entrer 1, 2, 3 ou 4.")
    
    def setup_complete(self):
        """Configuration complète"""
        print("\n🚀 CONFIGURATION COMPLÈTE")
        print("══════════════════════════")
        
        print("\n1. Installation des dépendances...")
        self.install_dependencies()
        
        print("\n2. Configuration du dataset...")
        self.setup_dataset()
        
        print("\n3. Configuration cloud...")
        self.setup_cloud_config()
        
        print("\n4. Installation locale...")
        self.setup_local()
        
        print("\n✅ Configuration complète terminée !")
        self.show_next_steps_complete()
    
    def setup_cloud_only(self):
        """Configuration cloud uniquement"""
        print("\n☁️  CONFIGURATION CLOUD")
        print("═══════════════════════")
        
        print("\n🤔 Quelle plateforme cloud utilisez-vous ?")
        print("1. Google Colab (gratuit)")
        print("2. AWS SageMaker") 
        print("3. Azure ML")
        print("4. Autre/Générique")
        
        platform_choice = input("\nVotre choix (1-4) : ").strip()
        
        platform_map = {
            '1': 'colab',
            '2': 'aws', 
            '3': 'azure',
            '4': 'generic'
        }
        
        platform = platform_map.get(platform_choice, 'generic')
        
        print(f"\n⚙️ Configuration pour {platform.upper()}...")
        self.setup_cloud_config(platform)
        
        print("\n📝 Création du dataset...")
        self.setup_dataset(cloud_only=True)
        
        print("\n✅ Configuration cloud terminée !")
        self.show_next_steps_cloud(platform)
    
    def setup_local_only(self):
        """Installation locale uniquement"""  
        print("\n🏠 INSTALLATION LOCALE")
        print("═══════════════════════")
        
        print("\n🤔 Avez-vous déjà un modèle entraîné ?")
        print("1. Oui, j'ai un modèle prêt")
        print("2. Non, je veux télécharger un modèle existant")
        print("3. Je veux utiliser le modèle de base")
        
        model_choice = input("\nVotre choix (1-3) : ").strip()
        
        if model_choice == '1':
            model_path = input("📁 Chemin vers votre modèle : ").strip()
            self.setup_local(existing_model=model_path)
        elif model_choice == '2':
            self.show_available_models()
            model_id = input("🔗 ID du modèle à télécharger : ").strip()
            self.setup_local(download_model=model_id)
        else:
            self.setup_local(use_base_model=True)
        
        print("\n✅ Installation locale terminée !")
        self.show_next_steps_local()
    
    def setup_advanced(self):
        """Configuration avancée personnalisée"""
        print("\n🔧 CONFIGURATION AVANCÉE")
        print("═══════════════════════════")
        
        config = {}
        
        # Configuration modèle
        print("\n📱 CONFIGURATION DU MODÈLE")
        config['model'] = self.configure_model_advanced()
        
        # Configuration performance
        print("\n⚡ CONFIGURATION PERFORMANCE") 
        config['performance'] = self.configure_performance_advanced()
        
        # Configuration sécurité
        print("\n🔒 CONFIGURATION SÉCURITÉ")
        config['security'] = self.configure_security_advanced()
        
        # Configuration interface
        print("\n🖥️  CONFIGURATION INTERFACE")
        config['interface'] = self.configure_interface_advanced()
        
        # Sauvegarder la configuration
        self.save_advanced_config(config)
        
        print("\n✅ Configuration avancée terminée !")
        self.show_next_steps_advanced()
    
    def install_dependencies(self):
        """Installe les dépendances nécessaires"""
        try:
            print("📦 Installation des dépendances Python...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            print("✅ Dépendances installées")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation dépendances: {e}")
            return False
        return True
    
    def setup_dataset(self, cloud_only=False):
        """Configure le dataset"""
        try:
            print("📊 Création du dataset initial...")
            subprocess.run([
                sys.executable, "scripts/dataset_manager.py", "--create-initial"
            ], check=True)
            
            if not cloud_only:
                enrich = input("🌐 Enrichir le dataset depuis les sources publiques ? (y/N) : ").strip().lower()
                if enrich == 'y':
                    print("🔄 Enrichissement du dataset...")
                    subprocess.run([
                        sys.executable, "scripts/dataset_manager.py", "--enrich-all"
                    ], check=True)
            
            print("✅ Dataset configuré")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur configuration dataset: {e}")
    
    def setup_cloud_config(self, platform='auto'):
        """Configure l'entraînement cloud"""
        config_file = self.project_dir / f"configs/cloud/{platform}_config.json"
        
        if config_file.exists():
            print(f"✅ Configuration {platform} trouvée")
        else:
            print(f"⚠️ Configuration {platform} non trouvée, utilisation générique")
            
        # Configuration Weights & Biases
        wandb_key = input("🔑 Clé Weights & Biases (optionnel, Entrée pour ignorer) : ").strip()
        if wandb_key:
            os.environ['WANDB_API_KEY'] = wandb_key
            print("✅ Clé W&B configurée")
    
    def setup_local(self, existing_model=None, download_model=None, use_base_model=False):
        """Configure l'installation locale"""
        try:
            cmd = [sys.executable, "local_deployment/local_installer.py"]
            
            if existing_model:
                cmd.extend(["--model-source", existing_model])
            elif download_model:
                cmd.extend(["--model-source", f"huggingface:{download_model}"])
            elif use_base_model:
                cmd.extend(["--model-source", "https://huggingface.co/meta-llama/Llama-2-7b-hf"])
            
            subprocess.run(cmd, check=True)
            print("✅ Installation locale terminée")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation locale: {e}")
    
    def show_available_models(self):
        """Affiche les modèles disponibles"""
        print("\n📋 MODÈLES DISPONIBLES :")
        print("1. llama-cybersec-base - Modèle de base optimisé")
        print("2. llama-cybersec-advanced - Version avancée") 
        print("3. llama-cybersec-quantized - Version rapide (quantifiée)")
        print("4. custom - URL personnalisée")
    
    def configure_model_advanced(self) -> Dict:
        """Configuration avancée du modèle"""
        config = {}
        
        # Modèle par défaut
        print("Modèle par défaut:")
        print("1. LLaMA-2-7B (recommandé)")
        print("2. LLaMA-2-13B (plus puissant)")
        print("3. Autre")
        
        choice = input("Choix (1-3) : ").strip()
        
        if choice == '1':
            config['default_model'] = "meta-llama/Llama-2-7b-hf"
        elif choice == '2':
            config['default_model'] = "meta-llama/Llama-2-13b-hf" 
        else:
            config['default_model'] = input("Nom du modèle : ").strip()
        
        # Format préféré
        print("\nFormat de modèle préféré:")
        print("1. GGUF (recommandé pour local)")
        print("2. Transformers (original)")
        
        format_choice = input("Choix (1-2) : ").strip()
        config['preferred_format'] = "gguf" if format_choice == '1' else "transformers"
        
        return config
    
    def configure_performance_advanced(self) -> Dict:
        """Configuration avancée des performances"""
        config = {}
        
        # GPU
        gpu_choice = input("Activer le support GPU ? (Y/n) : ").strip().lower()
        config['enable_gpu'] = gpu_choice != 'n'
        
        # Threads CPU
        cpu_threads = input("Nombre de threads CPU (0 pour auto) : ").strip()
        config['cpu_threads'] = int(cpu_threads) if cpu_threads.isdigit() else 0
        
        # Cache
        cache_size = input("Taille du cache en MB (512 par défaut) : ").strip()
        config['cache_size_mb'] = int(cache_size) if cache_size.isdigit() else 512
        
        return config
    
    def configure_security_advanced(self) -> Dict:
        """Configuration avancée de la sécurité"""
        config = {}
        
        # Validation des entrées
        validation = input("Activer la validation des entrées ? (Y/n) : ").strip().lower()
        config['input_validation'] = validation != 'n'
        
        # Longueur maximale
        max_length = input("Longueur maximale des entrées (4096 par défaut) : ").strip()
        config['max_input_length'] = int(max_length) if max_length.isdigit() else 4096
        
        # Filtrage de contenu
        content_filter = input("Activer le filtrage de contenu ? (Y/n) : ").strip().lower()
        config['content_filtering'] = content_filter != 'n'
        
        return config
    
    def configure_interface_advanced(self) -> Dict:
        """Configuration avancée de l'interface"""
        config = {}
        
        # Interface web
        web_ui = input("Activer l'interface web ? (Y/n) : ").strip().lower()
        config['web_ui_enabled'] = web_ui != 'n'
        
        if config['web_ui_enabled']:
            web_port = input("Port interface web (8080 par défaut) : ").strip()
            config['web_ui_port'] = int(web_port) if web_port.isdigit() else 8080
        
        # API REST
        api = input("Activer l'API REST ? (Y/n) : ").strip().lower()
        config['api_enabled'] = api != 'n'
        
        if config['api_enabled']:
            api_port = input("Port API (8081 par défaut) : ").strip()
            config['api_port'] = int(api_port) if api_port.isdigit() else 8081
        
        return config
    
    def save_advanced_config(self, config: Dict):
        """Sauvegarde la configuration avancée"""
        config_path = self.project_dir / "configs/local/custom_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Configuration sauvegardée : {config_path}")
    
    def show_next_steps_complete(self):
        """Affiche les prochaines étapes pour configuration complète"""
        print(f"""
🎉 CONFIGURATION COMPLÈTE TERMINÉE !

📋 PROCHAINES ÉTAPES :

1. ☁️  ENTRAÎNEMENT CLOUD :
   • Ouvrez Google Colab : https://colab.research.google.com/
   • Importez le notebook : cloud_training/colab_notebook.ipynb
   • Ou lancez : python cloud_training/cloud_trainer.py

2. 🏠 UTILISATION LOCALE :
   • Lancez l'interface : ./launch.sh (Linux/Mac) ou launch.bat (Windows)
   • Ou directement : python local_deployment/local_interface.py
   • Interface web : http://localhost:8080
   • API REST : http://localhost:8081

3. 📚 DOCUMENTATION :
   • Guide complet : docs/GUIDE_UTILISATION.md
   • API : docs/API.md

🆘 AIDE :
   • Tests : python tests/validate_pipeline.py
   • Support : https://github.com/LeZelote01/BYJY-LLM/issues
        """)
    
    def show_next_steps_cloud(self, platform: str):
        """Affiche les prochaines étapes pour cloud"""
        if platform == 'colab':
            print(f"""
☁️  ENTRAÎNEMENT GOOGLE COLAB

📋 PROCHAINES ÉTAPES :

1. Ouvrez Google Colab : https://colab.research.google.com/
2. Activez le GPU : Runtime → Change runtime type → GPU
3. Collez ce code dans une cellule :

```python
!git clone https://github.com/LeZelote01/BYJY-LLM.git
%cd BYJY-LLM
!pip install -r requirements.txt

# Entraînement
!python cloud_training/cloud_trainer.py \\
  --model-name "meta-llama/Llama-2-7b-hf" \\
  --dataset-path "./datasets/enriched_dataset.jsonl" \\
  --epochs 3
```

4. Une fois terminé, téléchargez le modèle et installez-le localement
            """)
        else:
            print(f"""
☁️  ENTRAÎNEMENT {platform.upper()}

📋 PROCHAINES ÉTAPES :

1. Configurez vos credentials cloud
2. Lancez l'entraînement :
   python cloud_training/cloud_trainer.py \\
     --model-name "meta-llama/Llama-2-7b-hf" \\
     --dataset-path "./datasets/enriched_dataset.jsonl"

3. Le modèle sera uploadé automatiquement vers votre stockage cloud
4. Installez-le localement une fois terminé
            """)
    
    def show_next_steps_local(self):
        """Affiche les prochaines étapes pour local"""
        print(f"""
🏠 INSTALLATION LOCALE TERMINÉE !

📋 UTILISATION :

1. 🚀 Lancement rapide :
   • Linux/Mac : ./launch.sh
   • Windows : launch.bat

2. 🌐 Interface web :
   • URL : http://localhost:8080
   • Chat interactif avec le modèle
   • Configuration visuelle

3. 🔌 API REST :
   • URL : http://localhost:8081
   • Documentation : http://localhost:8081/docs

4. 💻 Ligne de commande :
   python local_deployment/local_interface.py --help

🔧 GESTION DES MODÈLES :
   • Lister : python utils/model_manager.py --action list
   • Installer : python utils/model_manager.py --action install --model-id MODEL_ID
   • Mettre à jour : python utils/model_manager.py --action update --model-id MODEL_ID
        """)
    
    def show_next_steps_advanced(self):
        """Affiche les prochaines étapes pour configuration avancée"""
        print(f"""
🔧 CONFIGURATION AVANCÉE TERMINÉE !

📋 UTILISATION AVEC CONFIGURATION PERSONNALISÉE :

1. Lancement avec config personnalisée :
   python local_deployment/local_interface.py \\
     --config configs/local/custom_config.json

2. Les paramètres avancés sont maintenant actifs :
   • Performance optimisée selon vos spécifications
   • Sécurité configurée selon vos besoins
   • Interface personnalisée

3. Modification de la configuration :
   • Éditez : configs/local/custom_config.json
   • Redémarrez le service pour appliquer les changements

📚 DOCUMENTATION AVANCÉE :
   • Configuration : docs/CONFIGURATION.md
   • Développement : docs/DEVELOPMENT.md
        """)
    
    def run(self):
        """Lance la configuration interactive"""
        self.welcome()
        choice = self.get_user_choice()
        
        if choice == '1':
            self.setup_complete()
        elif choice == '2':
            self.setup_cloud_only()
        elif choice == '3':
            self.setup_local_only()
        elif choice == '4':
            self.setup_advanced()


def main():
    parser = argparse.ArgumentParser(description='Configuration rapide LLaMA CyberSec')
    parser.add_argument('--auto', choices=['complete', 'cloud', 'local'], 
                       help='Configuration automatique sans interaction')
    parser.add_argument('--platform', choices=['colab', 'aws', 'azure'], 
                       help='Plateforme cloud pour configuration automatique')
    
    args = parser.parse_args()
    
    setup = QuickSetup()
    
    if args.auto:
        if args.auto == 'complete':
            setup.setup_complete()
        elif args.auto == 'cloud':
            platform = args.platform or 'colab'
            setup.setup_cloud_config(platform)
        elif args.auto == 'local':
            setup.setup_local()
    else:
        setup.run()


if __name__ == "__main__":
    main()