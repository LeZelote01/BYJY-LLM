#!/usr/bin/env python3
"""
Script d'entraînement LoRA/QLoRA pour LLaMA-3-8B Cybersécurité
Utilise Hugging Face Transformers + PEFT + bitsandbytes
Optimisé pour des ressources limitées (20GB RAM)
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import bitsandbytes as bnb
import wandb

class LLaMACyberSecTrainer:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",  # Changez selon votre accès
                 dataset_path: str = "/app/datasets/enriched_dataset.jsonl",
                 output_dir: str = "/app/models/lora_weights",
                 use_4bit: bool = True,
                 use_nested_quant: bool = True):
        
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        
        # Créer les répertoires nécessaires
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration du logging
        self.setup_logging()
        
        # Vérifications système
        self.check_system_requirements()
        
        # Configuration de quantisation
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
        
        # Configuration LoRA
        self.lora_config = LoraConfig(
            r=16,  # Rank - équilibre entre performance et efficacité
            lora_alpha=32,  # Scaling parameter
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Paramètres d'entraînement
        self.training_config = {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1, 
            "gradient_accumulation_steps": 16,
            "warmup_steps": 100,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 25,
            "save_steps": 500,
            "eval_steps": 500,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "output_dir": str(self.output_dir),
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "ddp_find_unused_parameters": False,
            "group_by_length": True,
            "report_to": "wandb" if self.is_wandb_available() else None,
        }
    
    def setup_logging(self):
        """Configure le système de logging"""
        log_dir = Path("/app/logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_system_requirements(self):
        """Vérifie les prérequis système"""
        self.logger.info("Vérification des prérequis système...")
        
        # Vérifier CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.warning("CUDA non disponible - entraînement CPU uniquement")
        
        # Vérifier la RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / 1e9
            self.logger.info(f"RAM disponible: {ram_gb:.1f} GB")
            
            if ram_gb < 16:
                self.logger.warning("RAM faible - considérez d'ajuster batch_size")
        except ImportError:
            self.logger.warning("psutil non disponible - impossible de vérifier la RAM")
        
        # Vérifier le dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.dataset_path}")
    
    def is_wandb_available(self) -> bool:
        """Vérifie si Weights & Biases est disponible"""
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    def load_dataset(self) -> Dataset:
        """Charge et préprocess le dataset"""
        self.logger.info(f"Chargement du dataset: {self.dataset_path}")
        
        # Lire le fichier JSONL
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'instruction' in item and 'output' in item:
                        data.append(item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Ligne JSON invalide ignorée: {e}")
        
        self.logger.info(f"Dataset chargé: {len(data)} exemples")
        
        # Convertir en format Dataset Hugging Face
        dataset = Dataset.from_list(data)
        
        # Split train/validation (90/10)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        self.logger.info(f"Split: {len(dataset['train'])} train, {len(dataset['test'])} validation")
        
        return dataset
    
    def format_prompt(self, instruction: str, output: str = None) -> str:
        """Formate les prompts pour l'entraînement"""
        prompt_template = """### Instruction:
{}

### Response:
{}"""
        
        if output is None:
            return prompt_template.format(instruction, "")
        return prompt_template.format(instruction, output)
    
    def preprocess_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Préprocesse le dataset pour l'entraînement"""
        def tokenize_function(examples):
            # Formater les prompts
            prompts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                prompt = self.format_prompt(instruction, output)
                prompts.append(prompt)
            
            # Tokeniser
            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=2048,  # Ajustez selon votre GPU
                padding=False,
                return_tensors=None
            )
            
            # Ajouter les labels (copie des input_ids pour language modeling)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        self.logger.info("Tokenisation du dataset...")
        
        # Appliquer la tokenisation
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def load_model_and_tokenizer(self):
        """Charge le modèle et le tokenizer"""
        self.logger.info(f"Chargement du modèle: {self.model_name}")
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Ajouter le token de padding si nécessaire
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Charger le modèle avec quantisation
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.use_4bit else None,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Préparer le modèle pour l'entraînement quantifié
        if self.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Appliquer LoRA
        model = get_peft_model(model, self.lora_config)
        
        # Afficher les paramètres entraînables
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def setup_trainer(self, model, tokenizer, dataset):
        """Configure le trainer"""
        self.logger.info("Configuration du trainer...")
        
        # Arguments d'entraînement
        training_args = TrainingArguments(**self.training_config)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Pas de masked language modeling
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self):
        """Lance l'entraînement complet"""
        try:
            self.logger.info("=== DÉBUT DE L'ENTRAÎNEMENT ===")
            
            # Initialiser wandb si disponible
            if self.is_wandb_available():
                wandb.init(
                    project="llama-cybersec-finetuning",
                    config=self.training_config,
                    name=f"llama-cybersec-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            # 1. Charger le dataset
            dataset = self.load_dataset()
            
            # 2. Charger le modèle et tokenizer
            model, tokenizer = self.load_model_and_tokenizer()
            
            # 3. Préprocesser le dataset
            tokenized_dataset = self.preprocess_dataset(dataset, tokenizer)
            
            # 4. Configurer le trainer
            trainer = self.setup_trainer(model, tokenizer, tokenized_dataset)
            
            # 5. Sauvegarder la configuration
            self.save_training_config(tokenizer)
            
            # 6. Lancer l'entraînement
            self.logger.info("Début de l'entraînement...")
            trainer.train()
            
            # 7. Sauvegarder le modèle final
            self.logger.info("Sauvegarde du modèle...")
            trainer.save_model()
            trainer.save_state()
            
            # 8. Évaluation finale
            eval_results = trainer.evaluate()
            self.logger.info(f"Résultats d'évaluation: {eval_results}")
            
            # 9. Sauvegarde des métriques
            self.save_training_metrics(eval_results)
            
            self.logger.info("=== ENTRAÎNEMENT TERMINÉ ===")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            raise
        finally:
            if self.is_wandb_available():
                wandb.finish()
    
    def save_training_config(self, tokenizer):
        """Sauvegarde la configuration d'entraînement"""
        config = {
            "model_name": self.model_name,
            "dataset_path": str(self.dataset_path),
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": self.lora_config.target_modules,
                "lora_dropout": self.lora_config.lora_dropout,
            },
            "training_config": self.training_config,
            "timestamp": datetime.now().isoformat(),
            "tokenizer_info": {
                "vocab_size": tokenizer.vocab_size,
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
            }
        }
        
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration sauvegardée: {config_path}")
    
    def save_training_metrics(self, metrics: Dict[str, Any]):
        """Sauvegarde les métriques d'entraînement"""
        metrics_with_timestamp = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
        }
        
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_with_timestamp, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Métriques sauvegardées: {metrics_path}")
    
    def test_model(self, test_prompts: List[str] = None):
        """Test le modèle entraîné avec quelques exemples"""
        if test_prompts is None:
            test_prompts = [
                "Créer une règle YARA pour détecter un ransomware",
                "Analyser ce log Apache pour identifier une attaque",
                "Expliquer les techniques de lateral movement dans les APTs"
            ]
        
        self.logger.info("Test du modèle entraîné...")
        
        try:
            # Charger le modèle entraîné
            from peft import PeftModel
            
            # Charger le modèle de base
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config if self.use_4bit else None,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            # Charger les poids LoRA
            model = PeftModel.from_pretrained(base_model, str(self.output_dir))
            
            # Charger le tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Tester chaque prompt
            for i, prompt in enumerate(test_prompts):
                self.logger.info(f"Test {i+1}: {prompt}")
                
                formatted_prompt = self.format_prompt(prompt)
                inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extraire seulement la réponse (après "### Response:")
                if "### Response:" in response:
                    response = response.split("### Response:")[-1].strip()
                
                self.logger.info(f"Réponse: {response[:200]}...")
                print(f"\n=== Test {i+1} ===")
                print(f"Prompt: {prompt}")
                print(f"Réponse: {response}")
                print("-" * 50)
        
        except Exception as e:
            self.logger.error(f"Erreur lors du test: {e}")
    
    def estimate_training_time(self):
        """Estime le temps d'entraînement"""
        try:
            # Compter les exemples dans le dataset
            example_count = 0
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example_count += 1
            
            # Calculs approximatifs
            batch_size = self.training_config["per_device_train_batch_size"]
            grad_accum = self.training_config["gradient_accumulation_steps"]
            epochs = self.training_config["num_train_epochs"]
            
            effective_batch_size = batch_size * grad_accum
            steps_per_epoch = example_count // effective_batch_size
            total_steps = steps_per_epoch * epochs
            
            # Estimation du temps (très approximative)
            if torch.cuda.is_available():
                seconds_per_step = 2  # Avec GPU
            else:
                seconds_per_step = 10  # CPU seulement
            
            total_seconds = total_steps * seconds_per_step
            hours = total_seconds / 3600
            
            self.logger.info(f"=== ESTIMATION D'ENTRAÎNEMENT ===")
            self.logger.info(f"Exemples: {example_count}")
            self.logger.info(f"Steps par époque: {steps_per_epoch}")
            self.logger.info(f"Steps total: {total_steps}")
            self.logger.info(f"Temps estimé: {hours:.1f} heures")
            
        except Exception as e:
            self.logger.warning(f"Impossible d'estimer le temps: {e}")

def main():
    parser = argparse.ArgumentParser(description='Entraînement LoRA pour LLaMA-3-8B Cybersécurité')
    parser.add_argument('--model-name', default="meta-llama/Llama-2-7b-hf", 
                        help='Nom du modèle Hugging Face')
    parser.add_argument('--dataset-path', default="/app/datasets/enriched_dataset.jsonl",
                        help='Chemin vers le dataset JSONL')
    parser.add_argument('--output-dir', default="/app/models/lora_weights",
                        help='Répertoire de sortie pour les poids LoRA')
    parser.add_argument('--no-4bit', action='store_true', 
                        help='Désactiver la quantisation 4-bit')
    parser.add_argument('--test-only', action='store_true',
                        help='Tester le modèle sans entraîner')
    parser.add_argument('--estimate-time', action='store_true',
                        help='Estimer le temps d\'entraînement')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Nombre d\'époques')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Taille de batch par device')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Taux d\'apprentissage')
    
    args = parser.parse_args()
    
    # Initialiser le trainer
    trainer = LLaMACyberSecTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_4bit=not args.no_4bit
    )
    
    # Ajuster les paramètres depuis les arguments
    trainer.training_config.update({
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })
    
    if args.estimate_time:
        trainer.estimate_training_time()
    
    if args.test_only:
        trainer.test_model()
    else:
        # Lancer l'entraînement
        trainer.train()
        
        # Test après entraînement
        trainer.test_model()
    
    print("\n✅ Entraînement terminé!")
    print(f"📁 Poids LoRA sauvegardés dans: {args.output_dir}")
    print("🔧 Utilisez merge_and_quantize.py pour créer le modèle final")

if __name__ == "__main__":
    main()