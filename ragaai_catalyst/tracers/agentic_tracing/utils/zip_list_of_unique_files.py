import os
import hashlib
import zipfile
import re
import ast
import importlib.util
import json
import astor
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# Define the PackageUsageRemover class
class PackageUsageRemover(ast.NodeTransformer):
    def __init__(self, package_name):
        self.package_name = package_name
        self.imported_names = set()
    
    def visit_Import(self, node):
        filtered_names = []
        for name in node.names:
            if not name.name.startswith(self.package_name):
                filtered_names.append(name)
            else:
                self.imported_names.add(name.asname or name.name)
        
        if not filtered_names:
            return None
        node.names = filtered_names
        return node
    
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith(self.package_name):
            self.imported_names.update(n.asname or n.name for n in node.names)
            return None
        return node
    
    def visit_Assign(self, node):
        if self._uses_package(node.value):
            return None
        return node
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.imported_names:
            return None
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.imported_names:
                return None
        return node
    
    def _uses_package(self, node):
        if isinstance(node, ast.Name) and node.id in self.imported_names:
            return True
        if isinstance(node, ast.Call):
            return self._uses_package(node.func)
        if isinstance(node, ast.Attribute):
            return self._uses_package(node.value)
        return False

# Define the function to remove package code from a source code string
def remove_package_code(source_code: str, package_name: str) -> str:
    try:
        tree = ast.parse(source_code)
        transformer = PackageUsageRemover(package_name)
        modified_tree = transformer.visit(tree)
        modified_code = astor.to_source(modified_tree)
        return modified_code
    except Exception as e:
        raise Exception(f"Error processing source code: {str(e)}")

# TraceDependencyTracker class
class TraceDependencyTracker:
    def __init__(self, output_dir=None):
        self.tracked_files = set()
        self.python_imports = set()
        self.output_dir = output_dir or os.getcwd()

    def track_file_access(self, filepath):
        if os.path.exists(filepath):
            self.tracked_files.add(os.path.abspath(filepath))

    def find_config_files(self, content, base_path):
        patterns = [
            r'(?:open|read|load|with\s+open)\s*\([\'"]([^\'"]*\.(?:json|yaml|yml|txt|cfg|config|ini))[\'"]',
            r'(?:config|cfg|conf|settings|file|path)(?:_file|_path)?\s*=\s*[\'"]([^\'"]*\.(?:json|yaml|yml|txt|cfg|config|ini))[\'"]',
            r'[\'"]([^\'"]*\.txt)[\'"]',
            r'[\'"]([^\'"]*\.(?:yaml|yml))[\'"]',
            r'from\s+(\S+)\s+import',
            r'import\s+(\S+)'
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                filepath = match.group(1)
                if not os.path.isabs(filepath):
                    full_path = os.path.join(os.path.dirname(base_path), filepath)
                else:
                    full_path = filepath
                if os.path.exists(full_path):
                    self.track_file_access(full_path)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            self.find_config_files(f.read(), full_path)
                    except (UnicodeDecodeError, IOError):
                        pass

    def analyze_python_imports(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module_name = node.module
                    else:
                        for name in node.names:
                            module_name = name.name.split('.')[0]
                    try:
                        spec = importlib.util.find_spec(module_name)
                        if spec and spec.origin and not spec.origin.startswith(os.path.dirname(importlib.__file__)):
                            self.python_imports.add(spec.origin)
                    except (ImportError, AttributeError):
                        pass
        except Exception as e:
            print(f"Warning: Could not analyze imports in {filepath}: {str(e)}")

    def create_zip(self, filepaths):
        for filepath in filepaths:
            abs_path = os.path.abspath(filepath)
            self.track_file_access(abs_path)
            try:
                with open(abs_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.find_config_files(content, abs_path)
                if filepath.endswith('.py'):
                    self.analyze_python_imports(abs_path)
            except Exception as e:
                print(f"Warning: Could not process {filepath}: {str(e)}")

        self.tracked_files.update(self.python_imports)
        hash_contents = []
        for filepath in sorted(self.tracked_files):
            if 'env' in filepath:
                continue
            try:
                with open(filepath, 'rb') as file:
                    content = file.read()
                    if filepath.endswith('.py'):
                        # Temporarily remove raga_catalyst code for hash calculation
                        content = remove_package_code(content.decode('utf-8'), 'ragaai_catalyst').encode('utf-8')
                    hash_contents.append(content)
            except Exception as e:
                print(f"Warning: Could not read {filepath} for hash calculation: {str(e)}")

        combined_content = b''.join(hash_contents)
        hash_id = hashlib.sha256(combined_content).hexdigest()

        zip_filename = os.path.join(self.output_dir, f'{hash_id}.zip')
        common_path = [os.path.abspath(p) for p in self.tracked_files if 'env' not in p]

        if common_path!=[]:
            base_path = os.path.commonpath(common_path)
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in sorted(self.tracked_files):
                if 'env' in filepath:
                    continue
                try:
                    relative_path = os.path.relpath(filepath, base_path)
                    zipf.write(filepath, relative_path)
                    # logger.info(f"Added to zip: {relative_path}")
                except Exception as e:
                    print(f"Warning: Could not add {filepath} to zip: {str(e)}")

        return hash_id, zip_filename

# Main function for creating a zip of unique files
def zip_list_of_unique_files(filepaths, output_dir):
    tracker = TraceDependencyTracker(output_dir)
    return tracker.create_zip(filepaths)

# Example usage
if __name__ == "__main__":
    filepaths = ["script1.py", "script2.py"]
    hash_id, zip_path = zip_list_of_unique_files(filepaths)
    print(f"Created zip file: {zip_path}")
    print(f"Hash ID: {hash_id}")
