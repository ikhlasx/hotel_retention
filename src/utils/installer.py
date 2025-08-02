import subprocess
import sys
import os

def check_and_install_requirements():
    """
    Checks if required packages are installed and installs them if missing.
    """
    requirements_path = os.path.join(os.path.dirname(__file__), '..', '..', 'requirements.txt')
    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        return False
        
    try:
        # Run pip check to ensure dependencies are consistent
        subprocess.check_call([sys.executable, '-m', 'pip', 'check'])
        print("All requirements are satisfied.")
        return True
    except subprocess.CalledProcessError:
        print("Inconsistent environment. Installing packages from requirements.txt...")
        try:
            # Install packages from requirements.txt
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
            print("Successfully installed required packages.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to install packages from requirements.txt. {e}")
            return False
