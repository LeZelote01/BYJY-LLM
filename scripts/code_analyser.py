#!/usr/bin/env python3
"""
Analyseur de code multi-langages robuste et performant.
Détecte vulnérabilités, problèmes de qualité, génère des tests et métriques.
"""

import os
import sys
import json
import ast
import re
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

# Dépendances optionnelles
try:
    import esprima  # JavaScript
except ImportError:
    esprima = None

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
except ImportError:
    Language = Parser = None

class CodeAnalyzer:
    def __init__(self, project_path: str, config: Optional[Dict] = None):
        self.project_path = Path(project_path).resolve()
        self.config = self.default_config()
        if config:
            self.config.update(config)
        self.logger = self.setup_logging()
        self.parsers = self.setup_parsers()
        self.vulnerability_patterns = self.load_vulnerability_patterns()
        self.test_generators = self.setup_test_generators()
        self.analysis_results = {
            'project_info': {},
            'files_analyzed': [],
            'vulnerabilities': [],
            'code_quality': [],
            'tests_generated': [],
            'recommendations': [],
            'metrics': {},
            'dependencies': {}
        }

    def default_config(self) -> Dict[str, Any]:
        return {
            'supported_extensions': [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.php', '.java',
                '.c', '.cpp', '.cs', '.rb', '.go', '.rs', '.html',
                '.css', '.sql', '.json', '.xml', '.yaml', '.yml'
            ],
            'ignore_dirs': {
                'node_modules', '.git', '__pycache__', '.pytest_cache',
                'dist', 'build', '.next', 'vendor', 'target'
            },
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'generate_tests': True,
            'run_static_analysis': True,
            'check_dependencies': True,
            'analyze_security': True,
            'output_format': 'json'
        }

    def setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("CodeAnalyzer")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def setup_parsers(self) -> Dict[str, Any]:
        parsers = {}
        if Language and Parser:
            try:
                PY_LANGUAGE = Language(tspython.language(), "python")
                py_parser = Parser()
                py_parser.set_language(PY_LANGUAGE)
                parsers['python'] = py_parser
                JS_LANGUAGE = Language(tsjavascript.language(), "javascript")
                js_parser = Parser()
                js_parser.set_language(JS_LANGUAGE)
                parsers['javascript'] = js_parser
            except Exception as e:
                self.logger.warning(f"Tree-sitter non disponible: {e}")
        return parsers

    def load_vulnerability_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        # Patterns améliorés et plus stricts
        return {
            'python': {
                'sql_injection': [
                    r'execute\s*\(\s*["\'].*%.*["\']',
                    r'cursor\.execute\s*\(\s*f?["\'].*\{.*\}.*["\']',
                    r'\.format\s*\(.*\)',
                    r'%\s*\([^)]*\)\s*%'
                ],
                'command_injection': [
                    r'os\.system\s*\(',
                    r'subprocess\.(call|run|Popen)\s*\(',
                    r'eval\s*\(',
                    r'exec\s*\('
                ],
                'path_traversal': [
                    r'open\s*\(\s*[^,]*\+',
                    r'os\.path\.join\s*\([^)]*request',
                    r'\.\./',
                    r'%2e%2e%2f'
                ],
                'hardcoded_secrets': [
                    r'(password|api[_-]?key|secret|token)\s*=\s*["\'][^"\']{8,}["\']'
                ]
            },
            'javascript': {
                'xss': [
                    r'innerHTML\s*=',
                    r'document\.write\s*\(',
                    r'eval\s*\(',
                    r'dangerouslySetInnerHTML'
                ],
                'prototype_pollution': [
                    r'__proto__',
                    r'constructor\.prototype',
                    r'Object\.prototype'
                ],
                'insecure_random': [
                    r'Math\.random\s*\(\)',
                    r'new Date\(\)\.getTime\(\)'
                ]
            },
            'php': {
                'sql_injection': [
                    r'mysql_query\s*\(\s*["\'].*\$',
                    r'mysqli_query\s*\([^,]*,\s*["\'].*\$',
                    r'\$.*\..*["\'].*\$'
                ],
                'file_inclusion': [
                    r'include\s*\(\s*\$',
                    r'require\s*\(\s*\$',
                    r'include_once\s*\(\s*\$',
                    r'require_once\s*\(\s*\$'
                ]
            }
        }

    def setup_test_generators(self) -> Dict[str, Any]:
        return {
            'python': PythonTestGenerator(),
            'javascript': JavaScriptTestGenerator(),
            'php': PHPTestGenerator()
        }

    def analyze_project(self) -> Dict[str, Any]:
        self.logger.info(f"Analyse du projet: {self.project_path}")
        self.analysis_results['project_info'] = self.get_project_info()
        files_to_analyze = self.discover_files()
        for file_path in files_to_analyze:
            self.analyze_file(file_path)
        if self.config['check_dependencies']:
            self.analysis_results['dependencies'] = self.analyze_dependencies()
        self.analysis_results['metrics'] = self.calculate_metrics()
        self.analysis_results['recommendations'] = self.generate_recommendations()
        if self.config['generate_tests']:
            self.generate_tests()
        self.logger.info("Analyse terminée")
        return self.analysis_results

    def get_project_info(self) -> Dict[str, Any]:
        info = {
            'name': self.project_path.name,
            'path': str(self.project_path),
            'size': self.get_directory_size(),
            'languages': set(),
            'frameworks': set(),
            'analysis_date': datetime.now().isoformat()
        }
        config_files = {
            'package.json': 'Node.js',
            'requirements.txt': 'Python',
            'Pipfile': 'Python',
            'composer.json': 'PHP',
            'pom.xml': 'Java',
            'Cargo.toml': 'Rust',
            'go.mod': 'Go'
        }
        for config_file, language in config_files.items():
            if (self.project_path / config_file).exists():
                info['languages'].add(language)
        info['languages'] = list(info['languages'])
        return info

    def discover_files(self) -> List[Path]:
        files = []
        for file_path in self.project_path.rglob('*'):
            if self.should_analyze_file(file_path):
                files.append(file_path)
        self.logger.info(f"Fichiers à analyser: {len(files)}")
        return sorted(files)

    def should_analyze_file(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        if file_path.suffix not in self.config['supported_extensions']:
            return False
        if any(ignored in file_path.parts for ignored in self.config['ignore_dirs']):
            return False
        try:
            if file_path.stat().st_size > self.config['max_file_size']:
                return False
        except OSError:
            return False
        return True

    def analyze_file(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            file_info = {
                'path': str(file_path.relative_to(self.project_path)),
                'absolute_path': str(file_path),
                'size': len(content),
                'lines': content.count('\n') + 1,
                'language': self.detect_language(file_path),
                'hash': hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest(),
                'vulnerabilities': [],
                'quality_issues': [],
                'complexity': 0
            }
            language = file_info['language']
            if language == 'python':
                self.analyze_python_file(file_path, content, file_info)
            elif language in ['javascript', 'typescript']:
                self.analyze_javascript_file(file_path, content, file_info)
            elif language == 'php':
                self.analyze_php_file(file_path, content, file_info)
            else:
                self.analyze_generic_file(file_path, content, file_info)
            if self.config['analyze_security']:
                self.analyze_security_patterns(content, file_info)
            self.analysis_results['files_analyzed'].append(file_info)
        except Exception as e:
            self.logger.error(f"Erreur analyse {file_path}: {e}")

    def detect_language(self, file_path: Path) -> str:
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.php': 'php',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        return extension_map.get(file_path.suffix, 'unknown')

    def analyze_python_file(self, file_path: Path, content: str, file_info: Dict):
        try:
            tree = ast.parse(content)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            file_info['functions'] = len(functions)
            file_info['classes'] = len(classes)
            file_info['complexity'] = self.calculate_python_complexity(tree)
            dangerous_imports = {'eval', 'exec', 'input', '__import__'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            file_info['vulnerabilities'].append({
                                'type': 'dangerous_import',
                                'severity': 'medium',
                                'line': node.lineno,
                                'message': f"Import dangereux: {alias.name}"
                            })
            for func in functions:
                if not ast.get_docstring(func):
                    file_info['quality_issues'].append({
                        'type': 'missing_docstring',
                        'severity': 'low',
                        'line': func.lineno,
                        'message': f"Fonction '{func.name}' sans docstring"
                    })
        except SyntaxError as e:
            file_info['syntax_errors'] = [str(e)]
        except Exception as e:
            self.logger.warning(f"Erreur analyse Python {file_path}: {e}")

    def analyze_javascript_file(self, file_path: Path, content: str, file_info: Dict):
        try:
            if esprima:
                try:
                    esprima.parseScript(content, loc=True)
                    file_info['ast_valid'] = True
                except Exception as e:
                    file_info['syntax_errors'] = [str(e)]
            function_count = len(re.findall(r'function\s+\w+|=>\s*{|=\s*function', content))
            file_info['functions'] = function_count
            console_logs = [(m.start(), m.group()) for m in re.finditer(r'console\.(log|debug|info)', content)]
            for pos, match in console_logs:
                line_num = content[:pos].count('\n') + 1
                file_info['quality_issues'].append({
                    'type': 'console_statement',
                    'severity': 'low',
                    'line': line_num,
                    'message': f"Console statement trouvé: {match}"
                })
            var_declarations = [(m.start(), m.group()) for m in re.finditer(r'\bvar\s+\w+', content)]
            for pos, match in var_declarations:
                line_num = content[:pos].count('\n') + 1
                file_info['quality_issues'].append({
                    'type': 'var_declaration',
                    'severity': 'low',
                    'line': line_num,
                    'message': "Utiliser 'let' ou 'const' plutôt que 'var'"
                })
        except Exception as e:
            self.logger.warning(f"Erreur analyse JavaScript {file_path}: {e}")

    def analyze_php_file(self, file_path: Path, content: str, file_info: Dict):
        try:
            function_count = len(re.findall(r'function\s+\w+\s*\(', content))
            class_count = len(re.findall(r'class\s+\w+', content))
            file_info['functions'] = function_count
            file_info['classes'] = class_count
            echo_statements = [(m.start(), m.group()) for m in re.finditer(r'echo\s+\$|print\s+\$', content)]
            for pos, match in echo_statements:
                line_num = content[:pos].count('\n') + 1
                file_info['vulnerabilities'].append({
                    'type': 'xss_risk',
                    'severity': 'medium',
                    'line': line_num,
                    'message': "Echo/print direct d'une variable - risque XSS"
                })
            if '<?=' in content or '<? ' in content:
                file_info['quality_issues'].append({
                    'type': 'short_php_tags',
                    'severity': 'medium',
                    'line': 1,
                    'message': "Utilisation de balises PHP courtes (non recommandé)"
                })
        except Exception as e:
            self.logger.warning(f"Erreur analyse PHP {file_path}: {e}")

    def analyze_generic_file(self, file_path: Path, content: str, file_info: Dict):
        file_info['complexity'] = len(re.findall(r'[{}();]', content))
        todos = [(m.start(), m.group()) for m in re.finditer(r'(TODO|FIXME|XXX|HACK):', content, re.IGNORECASE)]
        for pos, match in todos:
            line_num = content[:pos].count('\n') + 1
            file_info['quality_issues'].append({
                'type': 'todo_comment',
                'severity': 'info',
                'line': line_num,
                'message': f"Commentaire trouvé: {match}"
            })

    def analyze_security_patterns(self, content: str, file_info: Dict):
        language = file_info['language']
        if language in self.vulnerability_patterns:
            patterns = self.vulnerability_patterns[language]
            for vuln_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = [(m.start(), m.group()) for m in re.finditer(pattern, content, re.IGNORECASE)]
                    for pos, match in matches:
                        line_num = content[:pos].count('\n') + 1
                        severity = 'critical' if vuln_type in ['hardcoded_secrets', 'sql_injection', 'command_injection'] else 'high' if vuln_type in ['xss', 'path_traversal'] else 'medium'
                        file_info['vulnerabilities'].append({
                            'type': vuln_type,
                            'severity': severity,
                            'line': line_num,
                            'message': f"Pattern vulnérable détecté: {match[:50]}...",
                            'pattern': pattern
                        })
                        self.analysis_results['vulnerabilities'].append({
                            'file': file_info['path'],
                            'type': vuln_type,
                            'severity': severity,
                            'line': line_num,
                            'message': f"Pattern vulnérable détecté: {match[:50]}...",
                        })

    def calculate_python_complexity(self, tree) -> int:
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.And, ast.Or)):
                complexity += 1
        return complexity

    def analyze_dependencies(self) -> Dict[str, Any]:
        return {
            'python': self.analyze_python_dependencies(),
            'javascript': self.analyze_javascript_dependencies(),
            'php': self.analyze_php_dependencies()
        }

    def analyze_python_dependencies(self) -> Optional[Dict]:
        req_files = ['requirements.txt', 'Pipfile', 'pyproject.toml']
        for req_file in req_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if req_file == 'requirements.txt':
                        deps = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith('#')]
                        return {
                            'file': req_file,
                            'dependencies': deps,
                            'count': len(deps)
                        }
                except Exception as e:
                    self.logger.warning(f"Erreur lecture {req_file}: {e}")
        return None

    def analyze_javascript_dependencies(self) -> Optional[Dict]:
        package_path = self.project_path / 'package.json'
        if package_path.exists():
            try:
                with open(package_path, 'r', encoding='utf-8', errors='ignore') as f:
                    package_data = json.load(f)
                deps = {}
                deps.update(package_data.get('dependencies', {}))
                deps.update(package_data.get('devDependencies', {}))
                return {
                    'file': 'package.json',
                    'dependencies': list(deps.keys()),
                    'count': len(deps),
                    'versions': deps
                }
            except Exception as e:
                self.logger.warning(f"Erreur lecture package.json: {e}")
        return None

    def analyze_php_dependencies(self) -> Optional[Dict]:
        composer_path = self.project_path / 'composer.json'
        if composer_path.exists():
            try:
                with open(composer_path, 'r', encoding='utf-8', errors='ignore') as f:
                    composer_data = json.load(f)
                deps = {}
                deps.update(composer_data.get('require', {}))
                deps.update(composer_data.get('require-dev', {}))
                return {
                    'file': 'composer.json',
                    'dependencies': list(deps.keys()),
                    'count': len(deps),
                    'versions': deps
                }
            except Exception as e:
                self.logger.warning(f"Erreur lecture composer.json: {e}")
        return None

    def calculate_metrics(self) -> Dict[str, Any]:
        files = self.analysis_results['files_analyzed']
        if not files:
            return {}
        total_lines = sum(f.get('lines', 0) for f in files)
        total_size = sum(f.get('size', 0) for f in files)
        total_functions = sum(f.get('functions', 0) for f in files)
        total_classes = sum(f.get('classes', 0) for f in files)
        vuln_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for vuln in self.analysis_results['vulnerabilities']:
            severity = vuln.get('severity', 'low')
            vuln_counts[severity] = vuln_counts.get(severity, 0) + 1
        languages = {}
        for file_info in files:
            lang = file_info.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        return {
            'total_files': len(files),
            'total_lines': total_lines,
            'total_size': total_size,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'average_file_size': total_size / len(files) if files else 0,
            'vulnerabilities_by_severity': vuln_counts,
            'total_vulnerabilities': sum(vuln_counts.values()),
            'languages_distribution': languages,
            'security_score': self.calculate_security_score(vuln_counts, len(files))
        }

    def calculate_security_score(self, vuln_counts: Dict, total_files: int) -> float:
        if total_files == 0:
            return 100.0
        weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}
        total_score = sum(count * weights[severity] for severity, count in vuln_counts.items())
        penalty = min(total_score / total_files * 10, 100)
        score = max(0, 100 - penalty)
        return round(score, 1)

    def generate_recommendations(self) -> List[Dict[str, str]]:
        recommendations = []
        metrics = self.analysis_results.get('metrics', {})
        vuln_counts = metrics.get('vulnerabilities_by_severity', {})
        if vuln_counts.get('critical', 0) > 0:
            recommendations.append({
                'category': 'security',
                'priority': 'critical',
                'title': 'Vulnérabilités critiques détectées',
                'description': f"{vuln_counts['critical']} vulnérabilités critiques trouvées. Correction immédiate requise.",
                'action': 'Examiner et corriger toutes les vulnérabilités critiques avant mise en production.'
            })
        if vuln_counts.get('high', 0) > 0:
            recommendations.append({
                'category': 'security',
                'priority': 'high',
                'title': 'Vulnérabilités importantes',
                'description': f"{vuln_counts['high']} vulnérabilités de niveau élevé trouvées.",
                'action': 'Planifier la correction de ces vulnérabilités dans les prochaines releases.'
            })
        avg_file_size = metrics.get('average_file_size', 0)
        if avg_file_size > 2000:
            recommendations.append({
                'category': 'maintainability',
                'priority': 'medium',
                'title': 'Fichiers volumineux',
                'description': f"Taille moyenne des fichiers: {avg_file_size:.0f} caractères. Recommandé: < 2000.",
                'action': 'Refactoriser les gros fichiers en modules plus petits.'
            })
        dependencies = self.analysis_results.get('dependencies', {})
        for lang, dep_info in dependencies.items():
            if dep_info and dep_info.get('count', 0) > 50:
                recommendations.append({
                    'category': 'dependencies',
                    'priority': 'medium',
                    'title': f'Nombreuses dépendances {lang}',
                    'description': f"{dep_info['count']} dépendances détectées.",
                    'action': 'Réviser et optimiser les dépendances pour réduire la surface d\'attaque.'
                })
        return recommendations

    def generate_tests(self):
        for file_info in self.analysis_results['files_analyzed']:
            language = file_info.get('language')
            if language in self.test_generators:
                generator = self.test_generators[language]
                tests = generator.generate_tests(file_info)
                if tests:
                    self.analysis_results['tests_generated'].extend(tests)

    def get_directory_size(self) -> int:
        total_size = 0
        for file_path in self.project_path.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    pass
        return total_size

    def export_results(self, output_path: Optional[str] = None, format: str = 'json'):
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"code_analysis_{timestamp}.{format}"
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        elif format == 'html':
            self.export_html_report(output_path)
        return output_path

    def export_html_report(self, output_path: str):
        """Génère un rapport HTML"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Rapport d'Analyse de Code</title>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mt-4">📊 Rapport d'Analyse de Code</h1>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5>Informations Générales</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Projet:</strong> {project_name}</p>
                        <p><strong>Chemin:</strong> {project_path}</p>
                        <p><strong>Date d'analyse:</strong> {analysis_date}</p>
                        <p><strong>Fichiers analysés:</strong> {total_files}</p>
                        <p><strong>Lignes de code:</strong> {total_lines:,}</p>
                        <p><strong>Score de sécurité:</strong> <span class="badge bg-{security_badge}">{security_score}/100</span></p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-danger text-white">
                        <h5>Vulnérabilités</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="vulnChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h5>Langages</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="langChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h5>Recommandations</h5>
                    </div>
                    <div class="card-body">
                        {recommendations_html}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Graphique vulnérabilités
        const vulnCtx = document.getElementById('vulnChart').getContext('2d');
        new Chart(vulnCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Critiques', 'Élevées', 'Moyennes', 'Faibles'],
                datasets: [{{
                    data: {vuln_data},
                    backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
                }}]
            }}
        }});
        
        // Graphique langages
        const langCtx = document.getElementById('langChart').getContext('2d');
        new Chart(langCtx, {{
            type: 'pie',
            data: {{
                labels: {lang_labels},
                datasets: [{{
                    data: {lang_data},
                    backgroundColor: ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1']
                }}]
            }}
        }});
    </script>
</body>
</html>
        """
        
        # Préparer les données pour le template
        metrics = self.analysis_results.get('metrics', {})
        project_info = self.analysis_results.get('project_info', {})
        
        security_score = metrics.get('security_score', 0)
        security_badge = 'success' if security_score >= 80 else 'warning' if security_score >= 60 else 'danger'
        
        vuln_counts = metrics.get('vulnerabilities_by_severity', {})
        vuln_data = [vuln_counts.get('critical', 0), vuln_counts.get('high', 0), 
                    vuln_counts.get('medium', 0), vuln_counts.get('low', 0)]
        
        lang_dist = metrics.get('languages_distribution', {})
        lang_labels = list(lang_dist.keys())
        lang_data = list(lang_dist.values())
        
        # Générer HTML des recommandations
        recommendations_html = ""
        for rec in self.analysis_results.get('recommendations', []):
            priority_class = 'danger' if rec['priority'] == 'critical' else 'warning' if rec['priority'] == 'high' else 'info'
            recommendations_html += f"""
            <div class="alert alert-{priority_class}">
                <h6>{rec['title']}</h6>
                <p>{rec['description']}</p>
                <small><strong>Action:</strong> {rec['action']}</small>
            </div>
            """
        
        # Remplir le template
        html_content = html_template.format(
            project_name=project_info.get('name', 'N/A'),
            project_path=project_info.get('path', 'N/A'),
            analysis_date=project_info.get('analysis_date', 'N/A'),
            total_files=metrics.get('total_files', 0),
            total_lines=metrics.get('total_lines', 0),
            security_score=security_score,
            security_badge=security_badge,
            vuln_data=vuln_data,
            lang_labels=lang_labels,
            lang_data=lang_data,
            recommendations_html=recommendations_html
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

# Générateurs de tests (identiques à la version précédente)
class BaseTestGenerator:
    def generate_tests(self, file_info: Dict) -> List[Dict]:
        return []

class PythonTestGenerator(BaseTestGenerator):
    def generate_tests(self, file_info: Dict) -> List[Dict]:
        tests = []
        file_path = Path(file_info['absolute_path'])
        test_content = f'''"""
Tests générés automatiquement pour {file_info['path']}
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from {file_path.stem} import *
except ImportError as e:
    pytest.skip(f"Impossible d'importer le module: {{e}}", allow_module_level=True)

class Test{file_path.stem.title()}:
    def test_module_imports(self):
        assert True

    def test_functions_exist(self):
        pass

    def test_no_syntax_errors(self):
        import ast
        with open(r"{file_info['absolute_path']}", "r") as f:
            try:
                ast.parse(f.read())
            except SyntaxError:
                pytest.fail("Erreurs de syntaxe détectées")
'''
        tests.append({
            'file': f"test_{file_path.stem}.py",
            'content': test_content,
            'type': 'pytest',
            'language': 'python'
        })
        return tests

class JavaScriptTestGenerator(BaseTestGenerator):
    def generate_tests(self, file_info: Dict) -> List[Dict]:
        tests = []
        file_path = Path(file_info['absolute_path'])
        test_content = f'''/**
 * Tests générés automatiquement pour {file_info['path']}
 */
describe('{file_path.stem}', () => {{
    test('should load without errors', () => {{
        expect(() => {{
            require('./{file_path.name}');
        }}).not.toThrow();
    }});
    test('should have valid syntax', () => {{
        const fs = require('fs');
        const content = fs.readFileSync('{file_info['absolute_path']}', 'utf8');
        expect(() => {{
            new Function(content);
        }}).not.toThrow();
    }});
}});
'''
        tests.append({
            'file': f"{file_path.stem}.test.js",
            'content': test_content,
            'type': 'jest',
            'language': 'javascript'
        })
        return tests

class PHPTestGenerator(BaseTestGenerator):
    def generate_tests(self, file_info: Dict) -> List[Dict]:
        tests = []
        file_path = Path(file_info['absolute_path'])
        test_content = f'''<?php
/**
 * Tests générés automatiquement pour {file_info['path']}
 */
use PHPUnit\\Framework\\TestCase;
class {file_path.stem.title()}Test extends TestCase
{{
    public function testFileExists()
    {{
        $this->assertFileExists('{file_info['absolute_path']}');
    }}
    public function testValidSyntax()
    {{
        $output = null;
        $return = null;
        exec('php -l {file_info['absolute_path']}', $output, $return);
        $this->assertEquals(0, $return, 'Erreurs de syntaxe détectées');
    }}
}}
'''
        tests.append({
            'file': f"{file_path.stem.title()}Test.php",
            'content': test_content,
            'type': 'phpunit',
            'language': 'php'
        })
        return tests

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyseur de code complet')
    parser.add_argument('project_path', help='Chemin vers le projet à analyser')
    parser.add_argument('--output', '-o', help='Fichier de sortie')
    parser.add_argument('--format', '-f', choices=['json', 'html'], default='json', help='Format de sortie')
    parser.add_argument('--no-tests', action='store_true', help='Ne pas générer de tests')
    parser.add_argument('--no-security', action='store_true', help='Ignorer l\'analyse de sécurité')
    args = parser.parse_args()
    config = {
        'generate_tests': not args.no_tests,
        'analyze_security': not args.no_security,
        'output_format': args.format
    }
    analyzer = CodeAnalyzer(args.project_path, config)
    results = analyzer.analyze_project()
    output_path = analyzer.export_results(args.output, args.format)
    metrics = results.get('metrics', {})
    print(f"🎯 Analyse terminée: {args.project_path}")
    print(f"📁 Fichiers analysés: {metrics.get('total_files', 0)}")
    print(f"📝 Lignes de code: {metrics.get('total_lines', 0):,}")
    print(f"🔒 Score de sécurité: {metrics.get('security_score', 0)}/100")
    vuln_counts = metrics.get('vulnerabilities_by_severity', {})
    total_vulns = sum(vuln_counts.values())
    print(f"⚠️  Vulnérabilités: {total_vulns}")
    if total_vulns > 0:
        print(f"   🔴 Critiques: {vuln_counts.get('critical', 0)}")
        print(f"   🟠 Élevées: {vuln_counts.get('high', 0)}")
        print(f"   🟡 Moyennes: {vuln_counts.get('medium', 0)}")
        print(f"   🟢 Faibles: {vuln_counts.get('low', 0)}")
    print(f"📄 Rapport généré: {output_path}")
    critical_recs = [r for r in results.get('recommendations', []) if r.get('priority') == 'critical']
    if critical_recs:
        print(f"🚨 ACTIONS CRITIQUES REQUISES:")
        for rec in critical_recs:
            print(f"   • {rec['title']}: {rec['action']}")

if __name__ == "__main__":
    main()
