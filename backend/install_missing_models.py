import argostranslate.package
import argostranslate.translate
import spacy
import os
import subprocess
import sys

def install_spacy_models():
    models = [
        "en_core_web_sm",
        "pl_core_news_sm",
        "it_core_news_sm",
        "de_core_news_sm",
        "es_core_news_sm",
        "nb_core_news_sm", # Norwegian Bokmål
        "uk_core_news_sm"  # Ukrainian
    ]
    
    python_exe = sys.executable
    for model in models:
        print(f"Checking spaCy model: {model}...")
        try:
            spacy.load(model)
            print(f"Model {model} already installed.")
        except OSError:
            print(f"Installing spaCy model: {model}...")
            subprocess.check_call([python_exe, "-m", "spacy", "download", model])

def install_argos_packages():
    print("Updating Argos Translate package index...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    target_languages = ["it", "es", "no", "sv", "de", "pl", "nb", "nn", "uk"]
    
    pairs_to_install = []
    for pkg in available_packages:
        src = pkg.from_code
        tgt = pkg.to_code
        if (src == "en" and tgt in target_languages) or (src in target_languages and tgt == "en"):
            pairs_to_install.append(pkg)
            
    print(f"Found {len(pairs_to_install)} Argos packages to install.")
    
    installed_packages = argostranslate.package.get_installed_packages()
    installed_names = [f"{p.from_code}->{p.to_code}" for p in installed_packages]

    for pkg in pairs_to_install:
        pkg_name = f"{pkg.from_code}->{pkg.to_code}"
        if pkg_name in installed_names:
            print(f"Argos package {pkg_name} already installed.")
            continue
            
        print(f"Installing Argos package: {pkg_name}...")
        try:
            pkg.install()
        except Exception as e:
            print(f"Error installing {pkg_name}: {e}")

if __name__ == "__main__":
    print("Starting model installation...")
    install_spacy_models()
    install_argos_packages()
    print("Model installation complete.")
