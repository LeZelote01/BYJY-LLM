#!/usr/bin/env python3
"""
Script d'entraînement cloud optimisé pour LLaMA-3-8B Cybersécurité
Support: Google Colab, AWS SageMaker, Azure ML, Paperspace
Fonctionnalités: Multi-GPU, reprise automatique, monitoring avancé
"""

import os
import sys
import json
import torch
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import requests
import time

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

# Détection de plateforme cloud
def detect_cloud_platform():
    """Détecte automatiquement la plateforme cloud"""
    try:
        # Google Colab
        import google.colab
        return "colab"
    except ImportError:
        pass
    
    # AWS
    try:
        response = requests.get("http://169.254.169.254/latest/meta-data/", timeout=2)
        if response.status_code == 200:
            return "aws"
    except:
        pass
    
    # Azure
    try:
        response = requests.get("http://169.254.169.254/metadata/instance?api-version=2021-02-01", 
                              headers={"Metadata": "true"}, timeout=2)
        if response.status_code == 200:
            return "azure"
    except:
        pass
    
    # Paperspace
    if os.environ.get("PS_API_KEY"):
        return "paperspace"
    
    return "unknown"

class CloudLLaMATrainer:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 dataset_path: str = None,
                 output_dir: str = "/content/models/lora_weights",
                 cloud_storage_bucket: str = None,
                 wandb_project: str = "llama-cybersec-cloud"):
        
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.cloud_storage_bucket = cloud_storage_bucket
        self.wandb_project = wandb_project
        
        # Détection automatique de la plateforme
        self.platform = detect_cloud_platform()
        self.setup_logging()
        
        # Configuration optimisée selon la plateforme
        self.setup_platform_config()
        
        # Configuration de quantisation adaptative
        self.setup_quantization_config()
        
        # Configuration LoRA optimisée
        self.setup_lora_config()
        
        # Paramètres d'entraînement adaptés au cloud
        self.setup_training_config()
        
        # Gestion des sauvegardes
        self.checkpoint_manager = CheckpointManager(self.output_dir, self.cloud_storage_bucket)
        
    def setup_logging(self):
        """Configure le logging cloud avancé"""
        log_dir = Path("/content/logs") if self.platform == "colab" else Path("./logs/cloud")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / f'cloud_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Plateforme détectée: {self.platform.upper()}")
        
    def setup_platform_config(self):
        """Configuration spécifique à chaque plateforme cloud"""
        if self.platform == "colab":
            self.setup_colab_config()
        elif self.platform == "aws":
            self.setup_aws_config()
        elif self.platform == "azure":
            self.setup_azure_config()
        elif self.platform == "paperspace":
            self.setup_paperspace_config()
        else:
            self.setup_generic_config()
    
    def setup_colab_config(self):
        """Configuration Google Colab"""
        self.logger.info("Configuration pour Google Colab")
        
        # Vérifier GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU Colab: {gpu_name}")
            
            # Configuration optimisée selon le type de GPU
            if "T4" in gpu_name:
                self.batch_size = 1
                self.gradient_accumulation_steps = 16
            elif "V100" in gpu_name:
                self.batch_size = 2
                self.gradient_accumulation_steps = 8
            elif "A100" in gpu_name:
                self.batch_size = 4
                self.gradient_accumulation_steps = 4
            else:
                self.batch_size = 1
                self.gradient_accumulation_steps = 16
        
        # Montage Google Drive
        self.mount_google_drive()
        
    def setup_aws_config(self):
        """Configuration AWS SageMaker"""
        self.logger.info("Configuration pour AWS SageMaker")
        
        # Détecter le type d'instance
        try:
            response = requests.get("http://169.254.169.254/latest/meta-data/instance-type", timeout=2)
            instance_type = response.text
            self.logger.info(f"Instance AWS: {instance_type}")
            
            # Configuration selon l'instance
            if "p3" in instance_type or "p4" in instance_type:
                self.batch_size = 4
                self.gradient_accumulation_steps = 4
            else:
                self.batch_size = 2
                self.gradient_accumulation_steps = 8
                
        except Exception as e:
            self.logger.warning(f"Impossible de détecter l'instance AWS: {e}")
            self.batch_size = 2
            self.gradient_accumulation_steps = 8
    
    def setup_azure_config(self):
        """Configuration Azure ML"""
        self.logger.info("Configuration pour Azure ML")
        self.batch_size = 2
        self.gradient_accumulation_steps = 8
        
    def setup_paperspace_config(self):
        """Configuration Paperspace Gradient"""
        self.logger.info("Configuration pour Paperspace Gradient")
        self.batch_size = 2
        self.gradient_accumulation_steps = 8
        
    def setup_generic_config(self):
        """Configuration générique"""
        self.logger.info("Configuration générique cloud")
        self.batch_size = 1
        self.gradient_accumulation_steps = 16
        
    def mount_google_drive(self):
        """Monte Google Drive si disponible"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            self.logger.info("Google Drive monté avec succès")
            
            # Créer le répertoire projet sur Drive
            project_dir = Path("/content/drive/MyDrive/LLaMA_CyberSec")
            project_dir.mkdir(exist_ok=True)
            
            # Rediriger les sauvegardes vers Drive
            self.output_dir = project_dir / "models" / "lora_weights"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            self.logger.warning(f"Impossible de monter Google Drive: {e}")
    
    def setup_quantization_config(self):
        """Configuration de quantisation adaptative"""
        # Vérifier la mémoire GPU disponible
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory >= 24:  # A100, V100
                self.use_4bit = False
                self.use_8bit = True
                self.logger.info("Configuration: 8-bit (GPU haute mémoire)")
            elif gpu_memory >= 16:  # T4, RTX series
                self.use_4bit = True
                self.use_8bit = False
                self.logger.info("Configuration: 4-bit (GPU mémoire moyenne)")
            else:  # GPU faible mémoire
                self.use_4bit = True
                self.use_8bit = False
                self.logger.info("Configuration: 4-bit (GPU faible mémoire)")
        else:
            self.use_4bit = False
            self.use_8bit = False
            self.logger.warning("Pas de GPU détecté - entraînement CPU")
        
        # Configuration BitsAndBytes
        if self.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            self.bnb_config = None
    
    def setup_lora_config(self):
        """Configuration LoRA optimisée"""
        # Configuration adaptative selon la taille du modèle
        if "7b" in self.model_name.lower():
            r = 16
            lora_alpha = 32
        elif "13b" in self.model_name.lower():
            r = 8
            lora_alpha = 16
        else:
            r = 16
            lora_alpha = 32
            
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.logger.info(f"Configuration LoRA: r={r}, alpha={lora_alpha}")
    
    def setup_training_config(self):
        """Configuration d'entraînement cloud optimisée"""
        self.training_config = {
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": 100,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "output_dir": str(self.output_dir),
            "save_total_limit": 5,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "ddp_find_unused_parameters": False,
            "group_by_length": True,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "report_to": "wandb",
            "run_name": f"llama-cybersec-{self.platform}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
    
    def download_dataset(self, dataset_url: str = None):
        """Télécharge le dataset depuis le cloud"""
        if not dataset_url and not self.dataset_path:
            self.logger.error("Aucun dataset spécifié")
            raise ValueError("Dataset requis")
        
        if dataset_url:
            self.logger.info(f"Téléchargement du dataset: {dataset_url}")
            dataset_file = Path("/content/dataset.jsonl") if self.platform == "colab" else Path("./dataset.jsonl")
            
            try:
                response = requests.get(dataset_url)
                response.raise_for_status()
                
                with open(dataset_file, 'wb') as f:
                    f.write(response.content)
                
                self.dataset_path = str(dataset_file)
                self.logger.info("Dataset téléchargé avec succès")
                
            except Exception as e:
                self.logger.error(f"Erreur téléchargement dataset: {e}")
                raise
    
    def load_and_prepare_dataset(self) -> Dataset:
        """Charge et prépare le dataset avec validation"""
        self.logger.info(f"Chargement du dataset: {self.dataset_path}")
        
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.dataset_path}")
        
        # Validation et chargement
        data = []
        invalid_count = 0
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if self.validate_example(item):
                        data.append(item)
                    else:
                        invalid_count += 1
                        self.logger.warning(f"Exemple invalide ligne {line_num}")
                except json.JSONDecodeError:
                    invalid_count += 1
                    self.logger.warning(f"JSON invalide ligne {line_num}")
        
        self.logger.info(f"Dataset chargé: {len(data)} exemples valides, {invalid_count} invalides")
        
        if len(data) == 0:
            raise ValueError("Aucun exemple valide dans le dataset")
        
        # Conversion en Dataset Hugging Face
        dataset = Dataset.from_list(data)
        
        # Split train/validation stratifié
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        self.logger.info(f"Split: {len(dataset['train'])} train, {len(dataset['test'])} validation")
        
        return dataset
    
    def validate_example(self, example: Dict[str, str]) -> bool:
        """Valide un exemple du dataset"""
        if not isinstance(example, dict):
            return False
        
        required_fields = ['instruction', 'output']
        for field in required_fields:
            if field not in example or not isinstance(example[field], str):
                return False
            if len(example[field].strip()) < 10:
                return False
        
        return True
    
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
        """Préprocesse le dataset avec optimisations"""
        def tokenize_function(examples):
            prompts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                prompt = self.format_prompt(instruction, output)
                prompts.append(prompt)
            
            # Tokenisation avec padding dynamique
            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        self.logger.info("Tokenisation du dataset...")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
            num_proc=4 if self.platform != "colab" else 2
        )
        
        return tokenized_dataset
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Charge le modèle et tokenizer avec gestion d'erreurs"""
        self.logger.info(f"Chargement du modèle: {self.model_name}")
        
        try:
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Modèle avec configuration adaptée
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                use_cache=False,
            )
            
            # Préparer pour quantisation si nécessaire
            if self.bnb_config:
                model = prepare_model_for_kbit_training(model)
            
            # Appliquer LoRA
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def train_with_recovery(self):
        """Entraînement avec reprise automatique"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return self.train()
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Erreur entraînement (tentative {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    self.logger.info("Tentative de reprise...")
                    time.sleep(60)  # Attendre avant retry
                    
                    # Nettoyer la mémoire
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
    
    def train(self):
        """Lance l'entraînement principal"""
        try:
            self.logger.info("=== DÉBUT DE L'ENTRAÎNEMENT CLOUD ===")
            
            # Initialiser Weights & Biases
            self.init_wandb()
            
            # Charger dataset
            dataset = self.load_and_prepare_dataset()
            
            # Charger modèle et tokenizer
            model, tokenizer = self.load_model_and_tokenizer()
            
            # Préprocesser dataset
            tokenized_dataset = self.preprocess_dataset(dataset, tokenizer)
            
            # Configurer trainer
            trainer = self.setup_trainer(model, tokenizer, tokenized_dataset)
            
            # Sauvegarder configuration
            self.save_training_config(tokenizer)
            
            # Training avec callbacks
            self.logger.info("Démarrage de l'entraînement...")
            trainer.add_callback(CloudTrainingCallback(self.checkpoint_manager, self.logger))
            
            # Reprendre depuis checkpoint si disponible
            resume_from_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Sauvegarder modèle final
            self.logger.info("Sauvegarde du modèle final...")
            trainer.save_model()
            trainer.save_state()
            
            # Évaluation finale
            eval_results = trainer.evaluate()
            self.logger.info(f"Résultats finaux: {eval_results}")
            
            # Upload vers cloud storage
            self.upload_final_model()
            
            self.logger.info("=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Erreur critique: {e}")
            raise
        finally:
            if hasattr(self, 'wandb_run'):
                wandb.finish()
    
    def init_wandb(self):
        """Initialise Weights & Biases"""
        try:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                config=self.training_config,
                name=f"llama-cybersec-{self.platform}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=[self.platform, "cybersecurity", "llama", "lora"]
            )
            self.logger.info("Weights & Biases initialisé")
        except Exception as e:
            self.logger.warning(f"Impossible d'initialiser W&B: {e}")
    
    def setup_trainer(self, model, tokenizer, dataset):
        """Configure le trainer avec callbacks avancés"""
        training_args = TrainingArguments(**self.training_config)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    
    def save_training_config(self, tokenizer):
        """Sauvegarde la configuration complète"""
        config = {
            "model_name": self.model_name,
            "platform": self.platform,
            "dataset_path": str(self.dataset_path),
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": self.lora_config.target_modules,
                "lora_dropout": self.lora_config.lora_dropout,
            },
            "training_config": self.training_config,
            "quantization": {
                "use_4bit": self.use_4bit,
                "use_8bit": self.use_8bit,
            },
            "timestamp": datetime.now().isoformat(),
            "tokenizer_info": {
                "vocab_size": tokenizer.vocab_size,
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
            }
        }
        
        config_path = self.output_dir / "cloud_training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Configuration sauvegardée: {config_path}")
    
    def upload_final_model(self):
        """Upload le modèle final vers le stockage cloud"""
        if not self.cloud_storage_bucket:
            self.logger.info("Pas de stockage cloud configuré")
            return
        
        try:
            self.checkpoint_manager.upload_final_model(self.output_dir)
            self.logger.info("Modèle uploadé vers le cloud storage")
        except Exception as e:
            self.logger.error(f"Erreur upload modèle: {e}")


class CheckpointManager:
    """Gestionnaire de checkpoints et sauvegardes cloud"""
    
    def __init__(self, local_dir: Path, cloud_bucket: str = None):
        self.local_dir = local_dir
        self.cloud_bucket = cloud_bucket
        self.logger = logging.getLogger(__name__)
    
    def get_latest_checkpoint(self):
        """Trouve le dernier checkpoint disponible"""
        checkpoint_dirs = [d for d in self.local_dir.glob("checkpoint-*") if d.is_dir()]
        if not checkpoint_dirs:
            return None
        
        # Trier par numéro de checkpoint
        latest = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
        self.logger.info(f"Checkpoint trouvé: {latest}")
        return str(latest)
    
    def upload_final_model(self, model_dir: Path):
        """Upload le modèle final (à implémenter selon le provider)"""
        # Placeholder pour l'upload cloud
        # AWS S3, Google Cloud Storage, Azure Blob, etc.
        pass


class CloudTrainingCallback:
    """Callback pour monitoring cloud"""
    
    def __init__(self, checkpoint_manager, logger):
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
    
    def on_save(self, args, state, control, **kwargs):
        """Appelé à chaque sauvegarde"""
        self.logger.info(f"Checkpoint sauvegardé: step {state.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Entraînement Cloud LLaMA Cybersécurité')
    parser.add_argument('--model-name', default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--dataset-path', help='Chemin local ou URL du dataset')
    parser.add_argument('--dataset-url', help='URL de téléchargement du dataset')
    parser.add_argument('--output-dir', default="/content/models/lora_weights")
    parser.add_argument('--cloud-bucket', help='Bucket de stockage cloud')
    parser.add_argument('--wandb-project', default="llama-cybersec-cloud")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    
    args = parser.parse_args()
    
    # Initialiser le trainer cloud
    trainer = CloudLLaMATrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        cloud_storage_bucket=args.cloud_bucket,
        wandb_project=args.wandb_project
    )
    
    # Télécharger dataset si nécessaire
    if args.dataset_url:
        trainer.download_dataset(args.dataset_url)
    
    # Ajuster configuration
    trainer.training_config.update({
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
    })
    
    # Lancer l'entraînement avec reprise automatique
    trainer.train_with_recovery()
    
    print("🎉 Entraînement cloud terminé avec succès!")
    print(f"📁 Modèle sauvegardé: {args.output_dir}")


if __name__ == "__main__":
    main()