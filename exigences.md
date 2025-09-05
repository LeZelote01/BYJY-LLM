Je veux que tu me crées de A à Z un projet complet pour obtenir un modèle LLaMA-3-8B personnalisé en cybersécurité et développement de code, que je pourrai ensuite exécuter localement sur ma machine (20 Go RAM, sans gros GPU).

### Étapes à réaliser :

1. Préparer un dataset JSONL initial (au moins 15 exemples) adapté à la cybersécurité et à l’analyse de code. 
   - Inclure des instructions et sorties sur : 
     - règles Sigma/YARA/Snort, 
     - détection d’attaques dans des logs, 
     - génération et analyse de scripts Bash/Python pour l’analyse réseau et projets de code, 
     - explications de concepts de pentest, DFIR, bonnes pratiques de développement, 
     - hypothèses d’erreurs et corrections de code.
   - Format attendu : {"instruction": "...", "output": "..."}

2. **Créer un script Python dataset_manager.py complet** qui permette de :  
   - Créer un dataset JSONL vide.  
   - Ajouter de nouveaux exemples manuellement.  
 Parser automatiquement toutes sortes de sources locales et distanteses** :  
       - Logs (syslog, Windows Event, Zeek/Suricata, PCAP converti en texte).  
       - Règles Sigma/YARA/Snort.  
       - Scripts Python ou Bash.  
       - Writeups CTF et fichiers texte génériques.  
       - Projets de code multi-fichiers et multi-langages.  
 Aller chercher automatiquement sur Internet toutes les sources publiques et légaleses** nécessaires pour enrichir le dataset :  
       - Repos GitHub open-source (Sigma rules, scripts, CTF writeups).  
       - Blogs et sites de sécurité publics, guides OWASP, MITRE ATT&CK, tutoriels.  
   - Transformer toutes ces sources en paires {"instruction": "...", "output": "..."}.  
   - Lister, modifier ou supprimer des entrées existantes.  
   - Vérifier la validité JSON de chaque ligne.  
   - Sauvegarder le dataset prêt pour l’entraînement LoRA/QLoRA.
Créer les scripts d’entraînementnt** (LoRA ou QLoRA) avec Hugging Face + PEFT + bitsandbytes. 
   - Fournir un fichier train_lora.py qui charge LLaMA-3-8B, applique QLoRA, entraîne sur le dataset JSONL, puis sauvegarde les poids LoRA.

4. **Inclure un script bash d’installation install.sh** pour un environnement cloud (Ubuntu 22.04) qui :
   - Installe Python 3.10+, CUDA/cuDNN si GPU Nvidia.
   - Clone les repos nécessaires (transformers, peft, bitsandbytes, etc.).
   - Télécharge LLaMA-3-8B depuis Hugging Face (instructions pour clé d’accès).
   - Lance l’entraînement avec train_lora.py.

5. **Le LLM final doit être capable de** :
   - Lire et analyser un projet complet multi-fichiers et multi-langages.  
   - Générer des tests unitaires et d’intégration pour chaque script.  
   - Détecter erreurs et mauvaises pratiques, puis proposer des corrections.  
   - rapport de validation completon complet** combinant l’analyse du code et les résultats de tests.  
   - Fournir les commandes ou instructions pour exécuter ces tests et validatienvironnement sandbox/VM/Docker/VM/Docker** séPréparer les commandes pour fusionner ou appliquer les poids LoRApoids LoRA** au modèle de base, puis exporter en format GGUF (quantifié en 4-bit ou 5-bit) utilisable avec llama.cpp ou Ollama.

7. **Préparer un guide pour installation locale** (Ubuntu 22.04, 20 Go RAM, CPU/GPU léger) :
   - Comment installer llama.cpp ou Ollama.
   - Comment placer le modèle quantifié et l’exécuter en inference.
   - Exemple de commandes pour analyser un projet complet, générer des tests et un rapport de validation.

### Contraintes :
- Tout doit être 100% automatisé et reproductible. 
- La sortie finale doit inclure : 
   - le dataset JSONL initial,
   - le script Python dataset_manager.py capable de parser et enrichir toutes les sources locales et Internet,
   - le script train_lora.py,
   - le script Bash install.sh,
   - les instructions de fusion/quantisation,
   - les étapes d’utilisation locale et du workflow de validation automatisée.

Livrer le tout dans un format bien structuré, comme si tu me donnais un package clé en main.
