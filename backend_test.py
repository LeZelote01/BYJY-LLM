#!/usr/bin/env python3
"""
Comprehensive Backend Testing Suite for BYJY-LLM Cybersecurity System
Tests all main components: dataset management, training scripts, interfaces, and utilities
"""

import os
import sys
import json
import subprocess
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BYJYLLMTester:
    def __init__(self):
        self.project_root = Path("/app")
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []
        
    def log_test_result(self, test_name, success, message, details=None):
        """Log test result"""
        status = "PASS" if success else "FAIL"
        logger.info(f"{status}: {test_name} - {message}")
        
        self.test_results[test_name] = {
            'success': success,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.passed_tests.append(test_name)
        else:
            self.failed_tests.append(test_name)
    
    def test_python_imports(self):
        """Test if all required Python modules can be imported"""
        logger.info("=== Testing Python Imports ===")
        
        required_modules = [
            'torch', 'transformers', 'datasets', 'peft', 'bitsandbytes',
            'flask', 'flask_cors', 'requests', 'bs4', 'git',
            'pandas', 'numpy', 'tqdm', 'psutil', 'yaml'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                missing_modules.append(f"{module}: {str(e)}")
        
        if not missing_modules:
            self.log_test_result("python_imports", True, "All required modules available")
        else:
            self.log_test_result("python_imports", False, f"Missing modules: {missing_modules}")
    
    def test_dataset_manager(self):
        """Test dataset_manager.py functionality"""
        logger.info("=== Testing Dataset Manager ===")
        
        try:
            # Test script execution with --help
            result = subprocess.run([
                sys.executable, str(self.project_root / "scripts" / "dataset_manager.py"), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_test_result("dataset_manager_help", True, "Help command works")
            else:
                self.log_test_result("dataset_manager_help", False, f"Help failed: {result.stderr}")
                return
            
            # Test importing the module
            sys.path.insert(0, str(self.project_root))
            from scripts.dataset_manager import DatasetManager
            
            # Test basic functionality
            with tempfile.TemporaryDirectory() as temp_dir:
                test_dataset_path = Path(temp_dir) / "test_dataset.jsonl"
                manager = DatasetManager(str(test_dataset_path))
                
                # Test dataset creation
                manager.create_initial_dataset()
                
                if test_dataset_path.exists():
                    self.log_test_result("dataset_creation", True, "Dataset creation works")
                    
                    # Test dataset validation
                    with open(test_dataset_path, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) >= 2:
                        # Validate JSON structure
                        valid_json = True
                        for line in lines:
                            try:
                                data = json.loads(line.strip())
                                if 'instruction' not in data or 'output' not in data:
                                    valid_json = False
                                    break
                            except json.JSONDecodeError:
                                valid_json = False
                                break
                        
                        if valid_json:
                            self.log_test_result("dataset_validation", True, f"Dataset valid with {len(lines)} examples")
                        else:
                            self.log_test_result("dataset_validation", False, "Invalid JSON structure in dataset")
                    else:
                        self.log_test_result("dataset_validation", False, f"Too few examples: {len(lines)}")
                else:
                    self.log_test_result("dataset_creation", False, "Dataset file not created")
                    
        except Exception as e:
            self.log_test_result("dataset_manager", False, f"Error: {str(e)}")
    
    def test_training_scripts(self):
        """Test training script functionality"""
        logger.info("=== Testing Training Scripts ===")
        
        # Test train_lora.py
        try:
            result = subprocess.run([
                sys.executable, str(self.project_root / "scripts" / "train_lora.py"), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_test_result("train_lora_help", True, "train_lora.py help works")
            else:
                self.log_test_result("train_lora_help", False, f"train_lora.py help failed: {result.stderr}")
        except Exception as e:
            self.log_test_result("train_lora_help", False, f"Error: {str(e)}")
        
        # Test cloud_trainer.py
        try:
            result = subprocess.run([
                sys.executable, str(self.project_root / "cloud_training" / "cloud_trainer.py"), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_test_result("cloud_trainer_help", True, "cloud_trainer.py help works")
            else:
                self.log_test_result("cloud_trainer_help", False, f"cloud_trainer.py help failed: {result.stderr}")
        except Exception as e:
            self.log_test_result("cloud_trainer_help", False, f"Error: {str(e)}")
    
    def test_model_utilities(self):
        """Test model management utilities"""
        logger.info("=== Testing Model Utilities ===")
        
        # Test model_manager.py
        try:
            result = subprocess.run([
                sys.executable, str(self.project_root / "utils" / "model_manager.py"), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_test_result("model_manager_help", True, "model_manager.py help works")
            else:
                self.log_test_result("model_manager_help", False, f"model_manager.py help failed: {result.stderr}")
        except Exception as e:
            self.log_test_result("model_manager_help", False, f"Error: {str(e)}")
        
        # Test model_manager list functionality
        try:
            result = subprocess.run([
                sys.executable, str(self.project_root / "utils" / "model_manager.py"), 
                "--action", "list"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_test_result("model_manager_list", True, "Model list functionality works")
            else:
                self.log_test_result("model_manager_list", False, f"Model list failed: {result.stderr}")
        except Exception as e:
            self.log_test_result("model_manager_list", False, f"Error: {str(e)}")
    
    def test_local_interface(self):
        """Test local interface functionality"""
        logger.info("=== Testing Local Interface ===")
        
        try:
            # Test importing the interface module
            sys.path.insert(0, str(self.project_root))
            from local_deployment.local_interface import ModernLLaMAInterface
            
            # Test basic initialization
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a minimal config for testing
                config_path = Path(temp_dir) / "test_config.json"
                test_config = {
                    "installation": {"install_dir": temp_dir},
                    "web_ui": {"host": "127.0.0.1", "port": 8080},
                    "model": {"name": "test-model"}
                }
                
                with open(config_path, 'w') as f:
                    json.dump(test_config, f)
                
                # Test interface initialization
                interface = ModernLLaMAInterface(config_path=str(config_path))
                
                missing_attrs = []
                if not hasattr(interface, 'app'):
                    missing_attrs.append('app')
                if not hasattr(interface, 'socketio'):
                    missing_attrs.append('socketio')
                
                if not missing_attrs:
                    self.log_test_result("local_interface_init", True, "Interface initialization successful")
                else:
                    self.log_test_result("local_interface_init", False, f"Interface missing: {missing_attrs}")
                    
        except Exception as e:
            self.log_test_result("local_interface_init", False, f"Error: {str(e)}")
    
    def test_local_installer(self):
        """Test local installer functionality"""
        logger.info("=== Testing Local Installer ===")
        
        try:
            # Test importing the installer module
            sys.path.insert(0, str(self.project_root))
            from local_deployment.local_installer import LocalModelInstaller
            
            # Test basic initialization
            with tempfile.TemporaryDirectory() as temp_dir:
                installer = LocalModelInstaller(install_dir=temp_dir)
                
                # Test system requirements check
                requirements = installer.check_system_requirements()
                
                if isinstance(requirements, bool):
                    self.log_test_result("installer_requirements", True, f"System requirements check completed: {requirements}")
                else:
                    self.log_test_result("installer_requirements", False, "Requirements check failed")
                    
        except Exception as e:
            self.log_test_result("local_installer", False, f"Error: {str(e)}")
    
    def test_configuration_files(self):
        """Test configuration file validity"""
        logger.info("=== Testing Configuration Files ===")
        
        config_file = self.project_root / "configs" / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check for required sections
                required_sections = ['model', 'training', 'lora', 'paths']
                missing_sections = [section for section in required_sections if section not in config]
                
                if not missing_sections:
                    self.log_test_result("config_validation", True, "Configuration file is valid")
                else:
                    self.log_test_result("config_validation", False, f"Missing sections: {missing_sections}")
                    
            except json.JSONDecodeError as e:
                self.log_test_result("config_validation", False, f"Invalid JSON: {str(e)}")
        else:
            self.log_test_result("config_validation", False, "Configuration file not found")
    
    def test_system_integration(self):
        """Test system integration using existing test suite"""
        logger.info("=== Testing System Integration ===")
        
        try:
            # Run the existing test system
            result = subprocess.run([
                sys.executable, str(self.project_root / "tests" / "test_system.py")
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log_test_result("system_integration", True, "System integration tests passed")
            else:
                self.log_test_result("system_integration", False, f"Integration tests failed: {result.stderr}")
                
        except Exception as e:
            self.log_test_result("system_integration", False, f"Error: {str(e)}")
    
    def test_pipeline_validation(self):
        """Test pipeline validation"""
        logger.info("=== Testing Pipeline Validation ===")
        
        try:
            # Run the pipeline validation with quick mode
            result = subprocess.run([
                sys.executable, str(self.project_root / "tests" / "validate_pipeline.py"), "--quick"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log_test_result("pipeline_validation", True, "Pipeline validation completed")
            else:
                self.log_test_result("pipeline_validation", False, f"Pipeline validation failed: {result.stderr}")
                
        except Exception as e:
            self.log_test_result("pipeline_validation", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("Starting comprehensive BYJY-LLM system testing...")
        
        # Run all test methods
        test_methods = [
            self.test_python_imports,
            self.test_configuration_files,
            self.test_dataset_manager,
            self.test_training_scripts,
            self.test_model_utilities,
            self.test_local_interface,
            self.test_local_installer,
            self.test_system_integration,
            self.test_pipeline_validation
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__
                self.log_test_result(test_name, False, f"Unexpected error: {str(e)}")
                logger.error(f"Error in {test_name}: {traceback.format_exc()}")
        
        # Generate summary
        return self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        logger.info(f"\n{'='*60}")
        logger.info("BYJY-LLM SYSTEM TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            logger.info(f"\nFailed Tests:")
            for test in self.failed_tests:
                result = self.test_results[test]
                logger.info(f"  ❌ {test}: {result['message']}")
        
        if self.passed_tests:
            logger.info(f"\nPassed Tests:")
            for test in self.passed_tests:
                logger.info(f"  ✅ {test}")
        
        return {
            'total': total_tests,
            'passed': passed_count,
            'failed': failed_count,
            'success_rate': (passed_count/total_tests)*100 if total_tests > 0 else 0,
            'failed_tests': self.failed_tests,
            'passed_tests': self.passed_tests,
            'detailed_results': self.test_results
        }

if __name__ == "__main__":
    tester = BYJYLLMTester()
    summary = tester.run_all_tests()
    
    # Exit with appropriate code
    if summary and summary['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)