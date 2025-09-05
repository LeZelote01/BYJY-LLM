# Système de Fine-tuning LLaMA-3-8B pour Cybersécurité

Ce projet fournit un système complet pour créer un modèle LLaMA-3-8B personnalisé spécialisé en cybersécurité et développement de code.

## Structure du Projet

```
/app/
├── datasets/                 # Datasets JSONL et sources
│   ├── initial_dataset.jsonl   # Dataset de base (15+ exemples)
│   ├── sources/                # Sources locales à parser
│   └── enriched_dataset.jsonl  # Dataset enrichi automatiquement
├── scripts/                  # Scripts d'entraînement et utilitaires
│   ├── dataset_manager.py       # Gestionnaire de dataset principal
│   ├── train_lora.py           # Script d'entraînement LoRA/QLoRA
│   ├── install.sh              # Installation automatisée
│   └── merge_and_quantize.py   # Fusion et quantisation
├── models/                   # Modèles et poids
│   ├── base/                   # Modèle LLaMA-3-8B de base
│   ├── lora_weights/           # Poids LoRA entraînés
│   └── quantized/              # Modèles quantifiés GGUF
├── configs/                  # Configurations d'entraînement
├── tests/                    # Tests et validation
└── docs/                     # Documentation et guides
```

## Fonctionnalités

### 1. Gestion de Dataset
- Création et enrichissement automatique de datasets JSONL
- Parsing de sources locales et distantes
- Support multi-formats (logs, règles, scripts, CTF writeups)

### 2. Fine-tuning Avancé
- Entraînement LoRA/QLoRA optimisé pour ressources limitées
- Support GPU/CPU avec quantisation intelligente
- Monitoring et checkpoints automatiques

### 3. Déploiement Local
- Quantisation GGUF 4-bit/5-bit
- Compatible llama.cpp et Ollama
- Optimisé pour 20 Go RAM, CPU/GPU léger

## Installation Rapide

```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

## Utilisation

1. **Créer le dataset initial** : `python scripts/dataset_manager.py --create-initial`
2. **Enrichir le dataset** : `python scripts/dataset_manager.py --enrich-all`
3. **Entraîner le modèle** : `python scripts/train_lora.py`
4. **Quantifier pour déploiement** : `python scripts/merge_and_quantize.py`

Voir `/docs/GUIDE_UTILISATION.md` pour les instructions détaillées.