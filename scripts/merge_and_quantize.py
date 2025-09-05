#!/usr/bin/env python3
"""
Script de fusion et quantisation pour LLaMA-3-8B Cybersécurité
Fusionne les poids LoRA avec le modèle de base et quantifie en GGUF
Compatible avec llama.cpp et Ollama
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ModelMergerQuantizer:
    def __init__(self,
                 base_model_name: str = "meta-llama/Llama-2-7b-hf",
                 lora_weights_path: str = "/app/models/lora_weights",
                 output_dir: str = "/app/models/quantized",
                 temp_dir: str = "/tmp/llama_merge"):
        
        self.base_model_name = base_model_name
        self.lora_weights_path = Path(lora_weights_path)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Créer les répertoires
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Chemins des outils externes
        self.llama_cpp_path = None
        self.find_llama_cpp()
    
    def setup_logging(self):
        """Configure le système de logging"""
        log_dir = Path("/app/logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'merge_quantize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_llama_cpp(self):
        """Trouve ou installe llama.cpp"""
        # Chercher llama.cpp dans les emplacements courants
        possible_paths = [
            "/usr/local/bin/llama-cpp",
            "/opt/llama.cpp",
            str(Path.home() / "llama.cpp"),
            "./llama.cpp"
        ]
        
        for path in possible_paths:
            convert_path = Path(path) / "convert.py"
            quantize_path = Path(path) / "quantize"
            
            if convert_path.exists() and quantize_path.exists():
                self.llama_cpp_path = Path(path)
                self.logger.info(f"llama.cpp trouvé: {self.llama_cpp_path}")
                return
        
        self.logger.warning("llama.cpp non trouvé - sera installé automatiquement")
    
    def install_llama_cpp(self):
        """Installe llama.cpp depuis GitHub"""
        self.logger.info("Installation de llama.cpp...")
        
        try:
            install_dir = Path.home() / "llama.cpp"
            
            # Cloner le repository
            if not install_dir.exists():
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(install_dir)
                ], check=True)
            else:
                # Mettre à jour si déjà présent
                subprocess.run([
                    "git", "-C", str(install_dir), "pull"
                ], check=True)
            
            # Compiler
            self.logger.info("Compilation de llama.cpp...")
            subprocess.run([
                "make", "-C", str(install_dir), "-j4"
            ], check=True)
            
            self.llama_cpp_path = install_dir
            self.logger.info(f"llama.cpp installé: {self.llama_cpp_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur lors de l'installation de llama.cpp: {e}")
            raise
    
    def verify_lora_weights(self) -> bool:
        """Vérifie que les poids LoRA existent et sont valides"""
        self.logger.info(f"Vérification des poids LoRA: {self.lora_weights_path}")
        
        if not self.lora_weights_path.exists():
            self.logger.error(f"Répertoire LoRA non trouvé: {self.lora_weights_path}")
            return False
        
        # Vérifier les fichiers essentiels
        required_files = [
            "adapter_config.json",
            "adapter_model.bin"  # ou "adapter_model.safetensors"
        ]
        
        for file in required_files:
            file_path = self.lora_weights_path / file
            safetensors_path = self.lora_weights_path / file.replace(".bin", ".safetensors")
            
            if not file_path.exists() and not safetensors_path.exists():
                self.logger.error(f"Fichier LoRA manquant: {file}")
                return False
        
        # Vérifier la configuration
        config_path = self.lora_weights_path / "adapter_config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Configuration LoRA: r={config.get('r')}, alpha={config.get('lora_alpha')}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture de la configuration LoRA: {e}")
            return False
        
        self.logger.info("Poids LoRA vérifiés avec succès")
        return True
    
    def merge_lora_weights(self) -> str:
        """Fusionne les poids LoRA avec le modèle de base"""
        self.logger.info("Fusion des poids LoRA avec le modèle de base...")
        
        if not self.verify_lora_weights():
            raise ValueError("Poids LoRA invalides")
        
        try:
            # Charger le tokenizer
            self.logger.info("Chargement du tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Charger le modèle de base
            self.logger.info(f"Chargement du modèle de base: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Charger et appliquer les poids LoRA
            self.logger.info("Application des poids LoRA...")
            model = PeftModel.from_pretrained(
                base_model, 
                str(self.lora_weights_path),
                torch_dtype=torch.float16
            )
            
            # Fusionner les poids
            self.logger.info("Fusion des poids...")
            merged_model = model.merge_and_unload()
            
            # Sauvegarder le modèle fusionné
            merged_path = self.temp_dir / "merged_model"
            merged_path.mkdir(exist_ok=True)
            
            self.logger.info(f"Sauvegarde du modèle fusionné: {merged_path}")
            merged_model.save_pretrained(
                str(merged_path),
                safe_serialization=True,
                max_shard_size="5GB"
            )
            tokenizer.save_pretrained(str(merged_path))
            
            # Nettoyer la mémoire
            del base_model, model, merged_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.logger.info("Fusion terminée avec succès")
            return str(merged_path)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la fusion: {e}")
            raise
    
    def convert_to_gguf(self, merged_model_path: str, quantization_type: str = "q4_0") -> str:
        """Convertit le modèle fusionné en format GGUF"""
        self.logger.info(f"Conversion en format GGUF (quantisation: {quantization_type})...")
        
        if self.llama_cpp_path is None:
            self.install_llama_cpp()
        
        try:
            # Chemin de sortie GGUF
            gguf_path = self.output_dir / f"llama-cybersec-{quantization_type}.gguf"
            temp_gguf_path = self.temp_dir / "model.gguf"
            
            # Script de conversion
            convert_script = self.llama_cpp_path / "convert.py"
            
            # Conversion en format GGUF (non quantifié)
            self.logger.info("Conversion en GGUF...")
            subprocess.run([
                sys.executable, str(convert_script),
                merged_model_path,
                "--outtype", "f16",
                "--outfile", str(temp_gguf_path)
            ], check=True)
            
            # Quantisation
            if quantization_type != "f16":
                self.logger.info(f"Quantisation en {quantization_type}...")
                quantize_bin = self.llama_cpp_path / "quantize"
                
                subprocess.run([
                    str(quantize_bin),
                    str(temp_gguf_path),
                    str(gguf_path),
                    quantization_type
                ], check=True)
                
                # Supprimer le fichier temporaire non quantifié
                temp_gguf_path.unlink()
            else:
                # Pas de quantisation, juste renommer
                temp_gguf_path.rename(gguf_path)
            
            # Vérifier la taille du fichier
            file_size_gb = gguf_path.stat().st_size / (1024**3)
            self.logger.info(f"Modèle GGUF créé: {gguf_path} ({file_size_gb:.1f} GB)")
            
            return str(gguf_path)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur lors de la conversion GGUF: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Erreur inattendue: {e}")
            raise
    
    def test_quantized_model(self, gguf_path: str):
        """Teste le modèle quantifié avec llama.cpp"""
        self.logger.info("Test du modèle quantifié...")
        
        try:
            # Utiliser l'exécutable main de llama.cpp pour tester
            main_bin = self.llama_cpp_path / "main"
            
            if not main_bin.exists():
                self.logger.warning("Binaire 'main' non trouvé - compilation requise")
                return
            
            # Test simple avec un prompt
            test_prompt = "### Instruction:\nCréer une règle YARA simple pour détecter un fichier suspect\n\n### Response:\n"
            
            self.logger.info("Exécution du test...")
            result = subprocess.run([
                str(main_bin),
                "-m", gguf_path,
                "-p", test_prompt,
                "-n", "100",  # Générer 100 tokens
                "--temp", "0.7",
                "--top-p", "0.9"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.logger.info("Test réussi - extrait de la réponse:")
                # Extraire seulement la partie après "### Response:"
                output = result.stdout
                if "### Response:" in output:
                    response = output.split("### Response:")[-1].strip()
                    self.logger.info(f"Réponse: {response[:200]}...")
                else:
                    self.logger.info(f"Sortie: {output[:200]}...")
            else:
                self.logger.warning(f"Test échoué: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Test interrompu - timeout")
        except Exception as e:
            self.logger.warning(f"Erreur lors du test: {e}")
    
    def create_ollama_modelfile(self, gguf_path: str):
        """Crée un Modelfile pour Ollama"""
        self.logger.info("Création du Modelfile pour Ollama...")
        
        modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"### Instruction:
{{{{ .Prompt }}}}

### Response:
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

SYSTEM \"\"\"Tu es un expert en cybersécurité spécialisé dans l'analyse de malware, la détection d'intrusion, et le développement de règles de sécurité. Tu peux créer et analyser des règles Sigma, YARA, Snort, analyser des logs de sécurité, identifier des vulnérabilités dans le code, et expliquer les techniques d'attaque et de défense. Réponds de manière précise et technique.\"\"\"
"""
        
        modelfile_path = self.output_dir / "Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        self.logger.info(f"Modelfile créé: {modelfile_path}")
        
        # Instructions pour utiliser avec Ollama
        instructions = f"""
=== UTILISATION AVEC OLLAMA ===

1. Installer Ollama (si pas déjà fait):
   curl -fsSL https://ollama.ai/install.sh | sh

2. Créer le modèle:
   ollama create llama-cybersec -f {modelfile_path}

3. Utiliser le modèle:
   ollama run llama-cybersec "Créer une règle YARA pour détecter un ransomware"

4. API REST:
   curl -X POST http://localhost:11434/api/generate \\
     -H "Content-Type: application/json" \\
     -d '{{
       "model": "llama-cybersec",
       "prompt": "Analyser ce log de sécurité...",
       "stream": false
     }}'
"""
        
        with open(self.output_dir / "ollama_instructions.txt", 'w') as f:
            f.write(instructions)
        
        self.logger.info("Instructions Ollama sauvegardées")
    
    def create_llama_cpp_script(self, gguf_path: str):
        """Crée un script pour utiliser le modèle avec llama.cpp"""
        script_content = f"""#!/bin/bash
# Script pour utiliser le modèle LLaMA Cybersécurité avec llama.cpp

GGUF_PATH="{gguf_path}"
LLAMA_CPP_PATH="{self.llama_cpp_path}"

# Vérifier que les fichiers existent
if [[ ! -f "$GGUF_PATH" ]]; then
    echo "Erreur: Modèle GGUF non trouvé: $GGUF_PATH"
    exit 1
fi

if [[ ! -f "$LLAMA_CPP_PATH/main" ]]; then
    echo "Erreur: llama.cpp non trouvé: $LLAMA_CPP_PATH"
    exit 1
fi

# Mode interactif par défaut
if [[ $# -eq 0 ]]; then
    echo "Démarrage du mode interactif..."
    "$LLAMA_CPP_PATH/main" \\
        -m "$GGUF_PATH" \\
        --temp 0.7 \\
        --top-p 0.9 \\
        --top-k 40 \\
        -n 512 \\
        --repeat-penalty 1.1 \\
        -i \\
        --in-prefix "### Instruction:\\n" \\
        --in-suffix "\\n\\n### Response:\\n"
else
    # Mode prompt unique
    PROMPT="### Instruction:\\n$*\\n\\n### Response:\\n"
    "$LLAMA_CPP_PATH/main" \\
        -m "$GGUF_PATH" \\
        --temp 0.7 \\
        --top-p 0.9 \\
        -n 256 \\
        -p "$PROMPT"
fi
"""
        
        script_path = self.output_dir / "run_llama_cybersec.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        self.logger.info(f"Script llama.cpp créé: {script_path}")
    
    def save_model_info(self, gguf_path: str, quantization_type: str):
        """Sauvegarde les informations du modèle"""
        model_info = {
            "model_name": "LLaMA-3-8B-Cybersecurity",
            "base_model": self.base_model_name,
            "lora_weights": str(self.lora_weights_path),
            "quantization": quantization_type,
            "gguf_path": gguf_path,
            "file_size_gb": Path(gguf_path).stat().st_size / (1024**3),
            "created_at": datetime.now().isoformat(),
            "recommended_settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 512,
                "context_length": 2048
            },
            "usage_examples": [
                "Créer une règle YARA pour détecter un trojan bancaire",
                "Analyser ces logs Apache pour identifier une attaque par injection SQL",
                "Expliquer les techniques d'évasion utilisées par le groupe APT29",
                "Générer un script Python pour parser des logs Suricata",
                "Créer une règle Sigma pour détecter du lateral movement"
            ]
        }
        
        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Informations du modèle sauvegardées: {info_path}")
    
    def cleanup_temp_files(self):
        """Nettoie les fichiers temporaires"""
        self.logger.info("Nettoyage des fichiers temporaires...")
        
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.logger.info("Fichiers temporaires supprimés")
        except Exception as e:
            self.logger.warning(f"Erreur lors du nettoyage: {e}")
    
    def run_full_pipeline(self, quantization_types: List[str] = None):
        """Exécute le pipeline complet de fusion et quantisation"""
        if quantization_types is None:
            quantization_types = ["q4_0", "q5_1"]  # Quantisations recommandées
        
        self.logger.info("=== DÉBUT DU PIPELINE DE FUSION ET QUANTISATION ===")
        
        try:
            # 1. Fusion des poids LoRA
            merged_model_path = self.merge_lora_weights()
            
            # 2. Conversion et quantisation pour chaque type demandé
            gguf_paths = []
            for quant_type in quantization_types:
                self.logger.info(f"Quantisation {quant_type}...")
                gguf_path = self.convert_to_gguf(merged_model_path, quant_type)
                gguf_paths.append((gguf_path, quant_type))
            
            # 3. Tests et création des scripts
            for gguf_path, quant_type in gguf_paths:
                self.test_quantized_model(gguf_path)
                self.save_model_info(gguf_path, quant_type)
            
            # 4. Création des fichiers d'utilisation (pour le premier modèle)
            if gguf_paths:
                main_gguf_path, main_quant_type = gguf_paths[0]
                self.create_ollama_modelfile(main_gguf_path)
                self.create_llama_cpp_script(main_gguf_path)
            
            # 5. Nettoyage
            self.cleanup_temp_files()
            
            self.logger.info("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")
            
            # Afficher le résumé
            self.show_completion_summary(gguf_paths)
            
        except Exception as e:
            self.logger.error(f"Erreur dans le pipeline: {e}")
            raise
    
    def show_completion_summary(self, gguf_paths: List[tuple]):
        """Affiche un résumé de completion"""
        print(f"\n{'='*60}")
        print("🎉 FUSION ET QUANTISATION TERMINÉES AVEC SUCCÈS!")
        print(f"{'='*60}")
        
        print(f"\n📁 Répertoire de sortie: {self.output_dir}")
        print(f"🎯 Modèles créés:")
        
        total_size = 0
        for gguf_path, quant_type in gguf_paths:
            size_gb = Path(gguf_path).stat().st_size / (1024**3)
            total_size += size_gb
            print(f"  • {Path(gguf_path).name} ({quant_type}) - {size_gb:.1f} GB")
        
        print(f"\n💾 Taille totale: {total_size:.1f} GB")
        
        print(f"\n🚀 UTILISATION:")
        print(f"📋 Ollama:")
        print(f"  ollama create llama-cybersec -f {self.output_dir}/Modelfile")
        print(f"  ollama run llama-cybersec")
        
        print(f"\n🔧 llama.cpp:")
        print(f"  {self.output_dir}/run_llama_cybersec.sh")
        
        print(f"\n📚 Documentation:")
        print(f"  • model_info.json - Informations détaillées")
        print(f"  • ollama_instructions.txt - Guide Ollama")
        
        print(f"\n✅ Votre modèle LLaMA Cybersécurité est prêt à l'emploi!")

def main():
    parser = argparse.ArgumentParser(description='Fusion et quantisation LLaMA Cybersécurité')
    parser.add_argument('--base-model', default="meta-llama/Llama-2-7b-hf",
                        help='Nom du modèle de base')
    parser.add_argument('--lora-weights', default="/app/models/lora_weights",
                        help='Chemin vers les poids LoRA')
    parser.add_argument('--output-dir', default="/app/models/quantized",
                        help='Répertoire de sortie')
    parser.add_argument('--quantization', nargs='+', 
                        default=["q4_0", "q5_1"],
                        choices=["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16"],
                        help='Types de quantisation')
    parser.add_argument('--merge-only', action='store_true',
                        help='Fusion uniquement, pas de quantisation')
    parser.add_argument('--test-only', action='store_true',
                        help='Tester un modèle existant')
    
    args = parser.parse_args()
    
    # Initialiser le merger/quantizer
    merger = ModelMergerQuantizer(
        base_model_name=args.base_model,
        lora_weights_path=args.lora_weights,
        output_dir=args.output_dir
    )
    
    if args.test_only:
        # Chercher un modèle GGUF existant à tester
        gguf_files = list(Path(args.output_dir).glob("*.gguf"))
        if gguf_files:
            merger.test_quantized_model(str(gguf_files[0]))
        else:
            print("Aucun modèle GGUF trouvé pour test")
    elif args.merge_only:
        # Fusion uniquement
        merged_path = merger.merge_lora_weights()
        print(f"Modèle fusionné sauvegardé: {merged_path}")
    else:
        # Pipeline complet
        merger.run_full_pipeline(args.quantization)
    
    print("\n✅ Opération terminée!")

if __name__ == "__main__":
    main()