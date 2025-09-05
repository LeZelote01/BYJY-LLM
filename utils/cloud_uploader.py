#!/usr/bin/env python3
"""
Utilitaire d'upload vers le cloud pour LLaMA-3-8B Cybersécurité
Support: AWS S3, Google Cloud Storage, Azure Blob, Hugging Face Hub
"""

import os
import sys
import json
import logging
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

class CloudUploader:
    def __init__(self, provider: str = "auto", credentials: Dict = None):
        self.provider = provider.lower()
        self.credentials = credentials or {}
        
        self.setup_logging()
        
        # Détecter le provider automatiquement si demandé
        if self.provider == "auto":
            self.provider = self.detect_provider()
        
        # Initialiser le client selon le provider
        self.client = self.init_client()
        
    def setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def detect_provider(self) -> str:
        """Détecte automatiquement le provider cloud"""
        # Variables d'environnement AWS
        if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
            return "aws"
        
        # Variables d'environnement GCP
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return "gcp"
        
        # Variables d'environnement Azure
        if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            return "azure"
        
        # Variables d'environnement Hugging Face
        if os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            return "huggingface"
        
        self.logger.warning("Aucun provider détecté, utilisation d'AWS par défaut")
        return "aws"
    
    def init_client(self):
        """Initialise le client cloud"""
        try:
            if self.provider == "aws":
                return self.init_aws_client()
            elif self.provider == "gcp":
                return self.init_gcp_client()
            elif self.provider == "azure":
                return self.init_azure_client()
            elif self.provider == "huggingface":
                return self.init_huggingface_client()
            else:
                raise ValueError(f"Provider non supporté: {self.provider}")
                
        except Exception as e:
            self.logger.error(f"Erreur initialisation client {self.provider}: {e}")
            raise
    
    def init_aws_client(self):
        """Initialise le client AWS S3"""
        try:
            import boto3
            
            # Configuration AWS
            session = boto3.Session(
                aws_access_key_id=self.credentials.get("aws_access_key_id"),
                aws_secret_access_key=self.credentials.get("aws_secret_access_key"),
                region_name=self.credentials.get("region", "us-east-1")
            )
            
            client = session.client('s3')
            self.logger.info("Client AWS S3 initialisé")
            return client
            
        except ImportError:
            self.logger.error("boto3 requis pour AWS S3")
            raise
        except Exception as e:
            self.logger.error(f"Erreur client AWS: {e}")
            raise
    
    def init_gcp_client(self):
        """Initialise le client Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            # Configuration GCP
            credentials_path = self.credentials.get("credentials_path") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            
            if credentials_path:
                client = storage.Client.from_service_account_json(credentials_path)
            else:
                client = storage.Client()
            
            self.logger.info("Client Google Cloud Storage initialisé")
            return client
            
        except ImportError:
            self.logger.error("google-cloud-storage requis pour GCP")
            raise
        except Exception as e:
            self.logger.error(f"Erreur client GCP: {e}")
            raise
    
    def init_azure_client(self):
        """Initialise le client Azure Blob Storage"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Configuration Azure
            connection_string = self.credentials.get("connection_string") or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            
            if not connection_string:
                raise ValueError("Chaîne de connexion Azure requise")
            
            client = BlobServiceClient.from_connection_string(connection_string)
            self.logger.info("Client Azure Blob Storage initialisé")
            return client
            
        except ImportError:
            self.logger.error("azure-storage-blob requis pour Azure")
            raise
        except Exception as e:
            self.logger.error(f"Erreur client Azure: {e}")
            raise
    
    def init_huggingface_client(self):
        """Initialise le client Hugging Face Hub"""
        try:
            from huggingface_hub import HfApi
            
            # Configuration Hugging Face
            token = self.credentials.get("token") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            
            client = HfApi(token=token)
            self.logger.info("Client Hugging Face Hub initialisé")
            return client
            
        except ImportError:
            self.logger.error("huggingface-hub requis pour Hugging Face")
            raise
        except Exception as e:
            self.logger.error(f"Erreur client Hugging Face: {e}")
            raise
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calcule le hash SHA256 d'un fichier"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def upload_file(self, 
                   local_path: Path, 
                   remote_path: str, 
                   bucket_name: str,
                   metadata: Dict = None) -> bool:
        """Upload un fichier vers le cloud"""
        
        if not local_path.exists():
            self.logger.error(f"Fichier local non trouvé: {local_path}")
            return False
        
        try:
            self.logger.info(f"Upload {local_path} vers {self.provider}://{bucket_name}/{remote_path}")
            
            # Calculer le hash du fichier
            file_hash = self.calculate_file_hash(local_path)
            
            # Métadonnées par défaut
            default_metadata = {
                "uploaded_at": datetime.now().isoformat(),
                "file_size": str(local_path.stat().st_size),
                "file_hash": file_hash,
                "content_type": mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
            }
            
            if metadata:
                default_metadata.update(metadata)
            
            # Upload selon le provider
            if self.provider == "aws":
                return self.upload_to_s3(local_path, remote_path, bucket_name, default_metadata)
            elif self.provider == "gcp":
                return self.upload_to_gcs(local_path, remote_path, bucket_name, default_metadata)
            elif self.provider == "azure":
                return self.upload_to_azure(local_path, remote_path, bucket_name, default_metadata)
            elif self.provider == "huggingface":
                return self.upload_to_huggingface(local_path, remote_path, bucket_name, default_metadata)
            else:
                self.logger.error(f"Provider non supporté: {self.provider}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur upload: {e}")
            return False
    
    def upload_to_s3(self, local_path: Path, remote_path: str, bucket_name: str, metadata: Dict) -> bool:
        """Upload vers AWS S3"""
        try:
            # Préparer les métadonnées S3
            s3_metadata = {k: v for k, v in metadata.items() if k not in ["content_type"]}
            
            # Upload
            self.client.upload_file(
                str(local_path),
                bucket_name,
                remote_path,
                ExtraArgs={
                    'ContentType': metadata["content_type"],
                    'Metadata': s3_metadata
                }
            )
            
            self.logger.info(f"Upload S3 réussi: s3://{bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur upload S3: {e}")
            return False
    
    def upload_to_gcs(self, local_path: Path, remote_path: str, bucket_name: str, metadata: Dict) -> bool:
        """Upload vers Google Cloud Storage"""
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(remote_path)
            
            # Préparer les métadonnées
            blob.metadata = {k: v for k, v in metadata.items() if k not in ["content_type"]}
            blob.content_type = metadata["content_type"]
            
            # Upload
            with open(local_path, 'rb') as f:
                blob.upload_from_file(f)
            
            self.logger.info(f"Upload GCS réussi: gs://{bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur upload GCS: {e}")
            return False
    
    def upload_to_azure(self, local_path: Path, remote_path: str, container_name: str, metadata: Dict) -> bool:
        """Upload vers Azure Blob Storage"""
        try:
            blob_client = self.client.get_blob_client(
                container=container_name,
                blob=remote_path
            )
            
            # Upload
            with open(local_path, 'rb') as f:
                blob_client.upload_blob(
                    f,
                    content_type=metadata["content_type"],
                    metadata={k: v for k, v in metadata.items() if k not in ["content_type"]},
                    overwrite=True
                )
            
            self.logger.info(f"Upload Azure réussi: {container_name}/{remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur upload Azure: {e}")
            return False
    
    def upload_to_huggingface(self, local_path: Path, remote_path: str, repo_id: str, metadata: Dict) -> bool:
        """Upload vers Hugging Face Hub"""
        try:
            # Upload du fichier
            self.client.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                commit_message=f"Upload {local_path.name}"
            )
            
            self.logger.info(f"Upload Hugging Face réussi: {repo_id}/{remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur upload Hugging Face: {e}")
            return False
    
    def upload_directory(self, 
                        local_dir: Path, 
                        remote_prefix: str, 
                        bucket_name: str,
                        include_patterns: List[str] = None,
                        exclude_patterns: List[str] = None) -> Dict[str, bool]:
        """Upload un répertoire complet"""
        
        if not local_dir.exists() or not local_dir.is_dir():
            self.logger.error(f"Répertoire local non trouvé: {local_dir}")
            return {}
        
        results = {}
        
        # Parcourir tous les fichiers
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Vérifier les patterns d'inclusion/exclusion
                relative_path = file_path.relative_to(local_dir)
                
                if self.should_include_file(str(relative_path), include_patterns, exclude_patterns):
                    remote_path = f"{remote_prefix}/{relative_path}".replace("\\", "/")
                    
                    success = self.upload_file(
                        file_path,
                        remote_path,
                        bucket_name,
                        metadata={"original_path": str(relative_path)}
                    )
                    
                    results[str(relative_path)] = success
        
        # Résumé
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"Upload terminé: {successful}/{total} fichiers réussis")
        
        return results
    
    def should_include_file(self, 
                           file_path: str, 
                           include_patterns: List[str] = None, 
                           exclude_patterns: List[str] = None) -> bool:
        """Détermine si un fichier doit être inclus"""
        import fnmatch
        
        # Vérifier les patterns d'exclusion
        if exclude_patterns:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    return False
        
        # Vérifier les patterns d'inclusion
        if include_patterns:
            for pattern in include_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    return True
            return False  # Aucun pattern d'inclusion ne correspond
        
        return True  # Inclure par défaut
    
    def create_upload_manifest(self, 
                              local_dir: Path,
                              upload_results: Dict[str, bool]) -> Path:
        """Crée un manifeste des fichiers uploadés"""
        
        manifest = {
            "upload_info": {
                "timestamp": datetime.now().isoformat(),
                "provider": self.provider,
                "local_directory": str(local_dir),
                "total_files": len(upload_results),
                "successful_uploads": sum(1 for success in upload_results.values() if success),
                "failed_uploads": sum(1 for success in upload_results.values() if not success)
            },
            "files": {}
        }
        
        for file_path, success in upload_results.items():
            full_path = local_dir / file_path
            
            if full_path.exists():
                manifest["files"][file_path] = {
                    "success": success,
                    "size": full_path.stat().st_size,
                    "hash": self.calculate_file_hash(full_path) if success else None,
                    "last_modified": datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
                }
        
        # Sauvegarder le manifeste
        manifest_path = local_dir / "upload_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Manifeste créé: {manifest_path}")
        return manifest_path
    
    def download_file(self, 
                     remote_path: str, 
                     local_path: Path, 
                     bucket_name: str) -> bool:
        """Télécharge un fichier depuis le cloud"""
        
        try:
            self.logger.info(f"Téléchargement {self.provider}://{bucket_name}/{remote_path} vers {local_path}")
            
            # Créer le répertoire parent si nécessaire
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Téléchargement selon le provider
            if self.provider == "aws":
                return self.download_from_s3(remote_path, local_path, bucket_name)
            elif self.provider == "gcp":
                return self.download_from_gcs(remote_path, local_path, bucket_name)
            elif self.provider == "azure":
                return self.download_from_azure(remote_path, local_path, bucket_name)
            elif self.provider == "huggingface":
                return self.download_from_huggingface(remote_path, local_path, bucket_name)
            else:
                self.logger.error(f"Provider non supporté: {self.provider}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur téléchargement: {e}")
            return False
    
    def download_from_s3(self, remote_path: str, local_path: Path, bucket_name: str) -> bool:
        """Télécharge depuis AWS S3"""
        try:
            self.client.download_file(bucket_name, remote_path, str(local_path))
            self.logger.info(f"Téléchargement S3 réussi: {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur téléchargement S3: {e}")
            return False
    
    def download_from_gcs(self, remote_path: str, local_path: Path, bucket_name: str) -> bool:
        """Télécharge depuis Google Cloud Storage"""
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(remote_path)
            blob.download_to_filename(str(local_path))
            self.logger.info(f"Téléchargement GCS réussi: {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur téléchargement GCS: {e}")
            return False
    
    def download_from_azure(self, remote_path: str, local_path: Path, container_name: str) -> bool:
        """Télécharge depuis Azure Blob Storage"""
        try:
            blob_client = self.client.get_blob_client(
                container=container_name,
                blob=remote_path
            )
            
            with open(local_path, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            self.logger.info(f"Téléchargement Azure réussi: {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur téléchargement Azure: {e}")
            return False
    
    def download_from_huggingface(self, remote_path: str, local_path: Path, repo_id: str) -> bool:
        """Télécharge depuis Hugging Face Hub"""
        try:
            from huggingface_hub import hf_hub_download
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False
            )
            
            # Renommer si nécessaire
            if Path(downloaded_path) != local_path:
                Path(downloaded_path).rename(local_path)
            
            self.logger.info(f"Téléchargement Hugging Face réussi: {local_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur téléchargement Hugging Face: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Uploader cloud pour LLaMA CyberSec')
    parser.add_argument('--provider', choices=['aws', 'gcp', 'azure', 'huggingface', 'auto'], 
                       default='auto', help='Provider cloud')
    parser.add_argument('--action', choices=['upload', 'download'], required=True,
                       help='Action à effectuer')
    parser.add_argument('--local-path', required=True, help='Chemin local')
    parser.add_argument('--remote-path', required=True, help='Chemin distant')
    parser.add_argument('--bucket', required=True, help='Nom du bucket/container/repo')
    parser.add_argument('--include', nargs='*', help='Patterns d\'inclusion')
    parser.add_argument('--exclude', nargs='*', help='Patterns d\'exclusion')
    
    args = parser.parse_args()
    
    try:
        # Initialiser l'uploader
        uploader = CloudUploader(provider=args.provider)
        
        local_path = Path(args.local_path)
        
        if args.action == 'upload':
            if local_path.is_file():
                # Upload d'un fichier
                success = uploader.upload_file(local_path, args.remote_path, args.bucket)
                print(f"Upload {'réussi' if success else 'échoué'}")
            elif local_path.is_dir():
                # Upload d'un répertoire
                results = uploader.upload_directory(
                    local_path, 
                    args.remote_path, 
                    args.bucket,
                    include_patterns=args.include,
                    exclude_patterns=args.exclude
                )
                uploader.create_upload_manifest(local_path, results)
                
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                print(f"Upload terminé: {successful}/{total} fichiers réussis")
            else:
                print(f"Chemin local non trouvé: {local_path}")
                sys.exit(1)
        
        elif args.action == 'download':
            # Téléchargement
            success = uploader.download_file(args.remote_path, local_path, args.bucket)
            print(f"Téléchargement {'réussi' if success else 'échoué'}")
    
    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()