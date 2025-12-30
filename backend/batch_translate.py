import json
import os
import argostranslate.translate
import argostranslate.package

def translate_text(text: str, from_code: str, to_code: str = "en"):
    # Normalize language codes
    argos_from = "nb" if from_code == "no" else from_code
    
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((x for x in installed_languages if x.code == argos_from), None)
        to_lang = next((x for x in installed_languages if x.code == to_code), None)
        
        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            if translation:
                return translation.translate(text)
    except Exception as e:
        print(f"Error during translation: {e}")
    return text

def main():
    # Adjusted path for running from root
    file_path = os.path.join("backend", "roots", "pl.json")
    if not os.path.exists(file_path):
        # Alternative path if running inside backend/
        file_path = os.path.join("roots", "pl.json")
        
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    updated = 0
    
    # Structure: POS -> category -> word -> details
    for pos, categories in data.items():
        if not isinstance(categories, dict): continue
        for cat, words in categories.items():
            if not isinstance(words, dict): continue
            for word, details in words.items():
                count += 1
                if not details.get("translation") or details["translation"] == "":
                    trans = translate_text(word, "pl", "en")
                    # Only update if translation is different from original (avoiding Argos fallbacks)
                    if trans.lower() != word.lower():
                        details["translation"] = trans
                        updated += 1
                        if updated % 10 == 0:
                            print(f"Progress: {updated} words translated...")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Finished. Scanned {count} words. Successfully translated {updated} missing entries.")

if __name__ == "__main__":
    main()
