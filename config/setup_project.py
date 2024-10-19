import sys
import os

# def setup_project_path():
#     # Get the current working directory (assuming you are in 'notebooks' directory)
#     notebook_dir = os.getcwd()

#     # Assuming the project structure is such that the src directory is one 
#     # level up from the notebook directory
#     project_root = os.path.dirname(os.path.dirname(notebook_dir))

#     # Add the project root to the Python path
#     if project_root not in sys.path:
#         sys.path.append(project_root)

#     # Also add the config directory to the Python path
#     config_path = os.path.join(project_root, 'config')
#     if config_path not in sys.path:
#         sys.path.append(config_path)

#     # Print the updated sys.path for verification
#     print("Project root added to sys.path:", project_root)
#     print("Config path added to sys.path:", config_path)
#     print("Current sys.path:", sys.path)
    
def setup_project_path():
    
    # The current file is located in project_root/config/setup_project.py
    config_path = os.path.dirname(__file__)
    project_root = os.path.dirname(config_path)
    
    sys.path.append(project_root)
    sys.path.append(config_path)
    
    

    # Print the updated sys.path for verification
    print("Project root added to sys.path:", project_root)
    print("Config path added to sys.path:", config_path)
    print("Current sys.path:", sys.path)
    
def add_src_path():
    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(path)