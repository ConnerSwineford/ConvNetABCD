import os
import subprocess

def install_packages():
    """
    Install required packages using pip.
    """
    packages = [
        'torch',
        'numpy',
        'pandas',
        'scipy',
        'nibabel',
        'tqdm'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def main():
    """
    Main function to set up the environment and install packages.
    """
    print("Setting up the environment and installing required packages...")

    install_packages()
    
    print("All packages installed successfully.")

if __name__ == "__main__":
    main()

