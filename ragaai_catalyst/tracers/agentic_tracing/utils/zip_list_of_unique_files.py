import os
import hashlib
import zipfile
import re
import ast
import importlib.util
import json
import ipynbname
import sys

from pathlib import Path
from IPython import get_ipython


if 'get_ipython' in locals():
    ipython_instance = get_ipython()
    if ipython_instance:
        ipython_instance.run_line_magic('reset', '-f')

import logging
logger = logging.getLogger(__name__)
logging_level = logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO


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
        remover = PackageUsageRemover(package_name)
        modified_tree = remover.visit(tree)
        modified_code = ast.unparse(modified_tree)

        return modified_code
    except Exception as e:
        logger.error(f"Error in remove_package_code: {e}")
        return source_code

class JupyterNotebookHandler:
    @staticmethod
    def is_running_in_colab():
        """Check if the code is running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_running_in_notebook():
        """Check if the code is running in a Jupyter notebook or Colab."""
        try:
            shell = get_ipython().__class__.__name__
            if JupyterNotebookHandler.is_running_in_colab():
                return True
            return shell == 'ZMQInteractiveShell'
        except:
            return False
    
    @staticmethod
    def get_notebook_path():
        """Get the path of the current executing notebook."""
        try:
            # First try using ipynbname
            try:
                notebook_path = ipynbname.path()
                if notebook_path:
                    # logger.info(f"Found notebook using ipynbname: {notebook_path}")
                    return str(notebook_path)
            except:
                pass

            # Check if running in Colab
            if JupyterNotebookHandler.is_running_in_colab():
                try:
                    from google.colab import drive
                    if not os.path.exists('/content/drive'):
                        drive.mount('/content/drive')
                        # logger.info("Google Drive mounted successfully")

                    # Look for notebooks in /content first
                    ipynb_files = list(Path('/content').glob('*.ipynb'))
                    if ipynb_files:
                        current_nb = max(ipynb_files, key=os.path.getmtime)
                        # logger.info(f"Found current Colab notebook: {current_nb}")
                        return str(current_nb)
                        
                    # Then check Drive if mounted
                    if os.path.exists('/content/drive'):
                        drive_ipynb_files = list(Path('/content/drive').rglob('*.ipynb'))
                        if drive_ipynb_files:
                            current_nb = max(drive_ipynb_files, key=os.path.getmtime)
                            # logger.info(f"Found Colab notebook in Drive: {current_nb}")
                            return str(current_nb)
                except Exception as e:
                    logger.warning(f"Error in Colab notebook detection: {str(e)}")

            # Try getting notebook path for regular Jupyter
            try:
                import IPython
                ipython = IPython.get_ipython()
                if ipython is not None:
                    # Try getting the notebook name from kernel
                    if hasattr(ipython, 'kernel') and hasattr(ipython.kernel, 'session'):
                        kernel_file = ipython.kernel.session.config.get('IPKernelApp', {}).get('connection_file', '')
                        if kernel_file:
                            kernel_id = Path(kernel_file).stem
                            current_dir = Path.cwd()
                            
                            # Look for .ipynb files in current and parent directories
                            for search_dir in [current_dir] + list(current_dir.parents):
                                notebooks = list(search_dir.glob('*.ipynb'))
                                recent_notebooks = [
                                    nb for nb in notebooks 
                                    if '.ipynb_checkpoints' not in str(nb)
                                ]
                                
                                if recent_notebooks:
                                    notebook_path = str(max(recent_notebooks, key=os.path.getmtime))
                                    # logger.info(f"Found Jupyter notebook: {notebook_path}")
                                    return notebook_path

                    # Try alternative method using notebook metadata
                    try:
                        notebook_path = ipython.kernel._parent_ident
                        if notebook_path:
                            # logger.info(f"Found notebook using kernel parent ident: {notebook_path}")
                            return notebook_path
                    except:
                        pass

            except Exception as e:
                # logger.warning(f"Error in Jupyter notebook detection: {str(e)}")
                return None
            
        except Exception as e:
            # logger.warning(f"Error getting notebook path: {str(e)}")
            return None



def comment_magic_commands(script_content: str) -> str:
    """Comment out magic commands, shell commands, and direct execution commands in the script content."""
    lines = script_content.splitlines()
    commented_lines = []
    for line in lines:
        # Check for magic commands, shell commands, or direct execution commands
        if re.match(r'^\s*(!|%|pip|apt-get|curl|conda)', line.strip()):
            commented_lines.append(f"# {line}")  # Comment the line
        else:
            commented_lines.append(line)  # Keep the line unchanged
    return "\n".join(commented_lines)



class TraceDependencyTracker:
    def __init__(self, output_dir=None):
        self.tracked_files = set()
        self.python_imports = set()
        self.notebook_path = None
        self.colab_content = None  
        
        # Set output directory with Colab handling
        if JupyterNotebookHandler.is_running_in_colab():
            self.output_dir = '/content'
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            logger.info("Using /content as output directory for Colab")
        else:
            self.output_dir = output_dir or os.getcwd()
        
        self.jupyter_handler = JupyterNotebookHandler()


    def check_environment_and_save(self):
        """Check if running in Colab and get current cell content."""
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if 'google.colab' in sys.modules:
                logger.info("Running on Google Colab.")
                
                # Retrieve the current cell content dynamically in Colab
                current_cell = ipython.history_manager.get_range()
                script_content = "\n".join(input_line for _, _, input_line in current_cell if input_line.strip())
                script_content = comment_magic_commands(script_content)  # Comment out magic commands
                
                # Store the content in the class attribute instead of saving to file
                self.colab_content = script_content
                logger.info("Successfully retrieved Colab cell content")
                
            else:
                logger.info("Not running on Google Colab.")
        except Exception as e:
            logger.warning(f"Error retrieving the current cell content: {e}")
        

    def track_jupyter_notebook(self):
        """Track the current notebook and its dependencies."""
        if self.jupyter_handler.is_running_in_notebook():
            # Get notebook path using the enhanced handler
            notebook_path = self.jupyter_handler.get_notebook_path()
            
            if notebook_path:
                self.notebook_path = notebook_path
                self.track_file_access(notebook_path)
                
                # Track notebook dependencies
                try:
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        notebook_content = f.read()
                        notebook_content = comment_magic_commands(notebook_content)
                        # Find and track imported files
                        self.find_config_files(notebook_content, notebook_path)
                except Exception as e:
                    pass
            else:
                pass


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
            pass

    def create_zip(self, filepaths):
        self.track_jupyter_notebook()
        # logger.info("Tracked Jupyter notebook and its dependencies")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # logger.info(f"Using output directory: {self.output_dir}")

        # Special handling for Colab
        if self.jupyter_handler.is_running_in_colab():
            # logger.info("Running in Google Colab environment")
            # Try to get the Colab notebook path
            colab_notebook = self.jupyter_handler.get_notebook_path()
            if colab_notebook:
                self.tracked_files.add(os.path.abspath(colab_notebook))
                # logger.info(f"Added Colab notebook to tracked files: {colab_notebook}")

            # Get current cell content
            self.check_environment_and_save()

        # Process all files (existing code)
        for filepath in filepaths:
            abs_path = os.path.abspath(filepath)
            self.track_file_access(abs_path)
            try:
                with open(abs_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Comment out magic commands before processing
                    content = comment_magic_commands(content)
                self.find_config_files(content, abs_path)
                if filepath.endswith('.py'):
                    self.analyze_python_imports(abs_path)
            except Exception as e:
                pass

        notebook_content_str = None
        if self.notebook_path and os.path.exists(self.notebook_path):
            try:
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    notebook_content = json.load(f)            
                            
                    cell_contents = []
                    for cell in notebook_content.get('cells', []):
                        if cell['cell_type'] == 'code':
                            # Comment out magic commands in the cell's source
                            cell_source = ''.join(cell['source'])
                            commented_source = comment_magic_commands(cell_source)
                            cell_contents.append(commented_source)

                    notebook_content_str = '\n\n'.join(cell_contents)
                    notebook_abs_path = os.path.abspath(self.notebook_path)
                    if notebook_abs_path in self.tracked_files:
                        self.tracked_files.remove(notebook_abs_path)

            except Exception as e:
                pass

        # Calculate hash and create zip
        self.tracked_files.update(self.python_imports)
        hash_contents = []

        for filepath in sorted(self.tracked_files):
            if not filepath.endswith('.py'):
                continue
            elif '/envs' in filepath or '__init__' in filepath:
                continue
            try:
                with open(filepath, 'rb') as file:
                    content = file.read()
                    content = remove_package_code(content.decode('utf-8'), 'ragaai_catalyst').encode('utf-8')
                    hash_contents.append(content)
            except Exception as e:
                logger.warning(f"Could not read {filepath} for hash calculation: {str(e)}")
                pass


        if notebook_content_str:
            hash_contents.append(notebook_content_str.encode('utf-8'))

        if self.colab_content:
            hash_contents.append(self.colab_content.encode('utf-8'))


        combined_content = b''.join(hash_contents)
        hash_id = hashlib.sha256(combined_content).hexdigest()

        # Create zip in the appropriate location
        zip_filename = os.path.join(self.output_dir, f'{hash_id}.zip')
        common_path = [os.path.abspath(p) for p in self.tracked_files if 'env' not in p]

        if common_path:
            base_path = os.path.commonpath(common_path)
        else:
            base_path = os.getcwd()

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in sorted(self.tracked_files):
                if 'env' in filepath or 'ragaai_catalyst' in filepath:
                    continue
                try:
                    relative_path = os.path.relpath(filepath, base_path)
                    zipf.write(filepath, relative_path)
                    logger.debug(f"Added python script to zip: {relative_path}")
                except Exception as e:
                    pass

            if notebook_content_str:
                py_filename = os.path.splitext(os.path.basename(self.notebook_path))[0] + ".py"
                zipf.writestr(py_filename, notebook_content_str)
                logger.debug(f"Added notebook content to zip as: {py_filename}")

            if self.colab_content:
                colab_filename = "colab_file.py"
                zipf.writestr(colab_filename, self.colab_content)
                logger.debug(f"Added Colab cell content to zip as: {colab_filename}")


        logger.info(" Zip file created successfully.")
        logger.debug(f"Zip file created successfully at: {zip_filename}")
        return hash_id, zip_filename

def zip_list_of_unique_files(filepaths, output_dir=None):
    """Create a zip file containing all unique files and their dependencies."""
    if output_dir is None:
        # Set default output directory based on environment
        if JupyterNotebookHandler.is_running_in_colab():
            output_dir = '/content'
        else:
            output_dir = os.getcwd()
    
    tracker = TraceDependencyTracker(output_dir)
    return tracker.create_zip(filepaths)


# Example usage
if __name__ == "__main__":
    filepaths = ["script1.py", "script2.py"]
    hash_id, zip_path = zip_list_of_unique_files(filepaths)
    print(f"Created zip file: {zip_path}")
    print(f"Hash ID: {hash_id}")
