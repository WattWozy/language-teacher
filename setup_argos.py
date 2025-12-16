import argostranslate.package
import argostranslate.translate

def install_languages():
    print("Updating package index...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    # Languages to support: Italian, Spanish, Norwegian, Swedish, English, German, Polish
    # We need pairs to/from English for all of these to support pivoting.
    # Codes: it, es, no (or nb?), sv, en, de, pl
    
    target_languages = ["it", "es", "no", "sv", "de", "pl", "nb", "nn"] # Added nb/nn for Norwegian just in case
    
    pairs_to_install = []
    
    for pkg in available_packages:
        src = pkg.from_code
        tgt = pkg.to_code
        
        # We want English <-> Target
        if src == "en" and tgt in target_languages:
            pairs_to_install.append(pkg)
        elif src in target_languages and tgt == "en":
            pairs_to_install.append(pkg)
            
    print(f"Found {len(pairs_to_install)} packages to install.")
    
    for pkg in pairs_to_install:
        print(f"Installing {pkg.from_code} -> {pkg.to_code}...")
        pkg.install()
        
    print("Installation complete.")

if __name__ == "__main__":
    install_languages()
