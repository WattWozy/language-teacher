import argostranslate.package
import argostranslate.translate

def test_translation():
    print("Testing Argos Translate...")
    
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        print(f"Installed languages: {[lang.code for lang in installed_languages]}")
        
        from_code = "pl"
        to_code = "en"
        text = "Cześć, jak się masz?"
        
        from_lang = next((x for x in installed_languages if x.code == from_code), None)
        to_lang = next((x for x in installed_languages if x.code == to_code), None)
        
        if from_lang and to_lang:
            print(f"Found languages: {from_lang.code} -> {to_lang.code}")
            translation = from_lang.get_translation(to_lang)
            if translation:
                result = translation.translate(text)
                print(f"Original: {text}")
                print(f"Translated: {result}")
            else:
                print("No translation path found.")
        else:
            print(f"Could not find languages. From: {from_lang}, To: {to_lang}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_translation()
