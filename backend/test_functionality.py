import sys
import os

# Add venv to path if needed (though running with venv python is better)
# For this script we assume it's run via .\backend\venv\Scripts\python.exe

import spacy
import argostranslate.translate
import argostranslate.package

def test():
    print("--- Testing spaCy ---")
    
    # Polish test
    try:
        nlp_pl = spacy.load("pl_core_news_sm")
        print("Successfully loaded pl_core_news_sm")
        doc = nlp_pl("jest")
        print(f"Result for 'jest': {doc[0].text} -> lemma: {doc[0].lemma_}, pos: {doc[0].pos_}")
    except Exception as e:
        print(f"Failed to load pl_core_news_sm: {e}")

    # English test
    try:
        nlp_en = spacy.load("en_core_web_sm")
        print("Successfully loaded en_core_web_sm")
        doc = nlp_en("running")
        print(f"Result for 'running': {doc[0].text} -> lemma: {doc[0].lemma_}, pos: {doc[0].pos_}")
    except Exception as e:
        print(f"Failed to load en_core_web_sm: {e}")

    print("\n--- Testing Argos Translate ---")
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        print(f"Installed language codes: {[lang.code for lang in installed_languages]}")
        
        for from_code in ["pl", "it", "no"]:
            to_code = "en"
            from_lang = next((x for x in installed_languages if x.code == from_code), None)
            to_lang = next((x for x in installed_languages if x.code == to_code), None)
            
            if from_lang and to_lang:
                translation = from_lang.get_translation(to_lang)
                if translation:
                    test_word = {"pl": "cześć", "it": "ciao", "no": "hei"}.get(from_code, "test")
                    result = translation.translate(test_word)
                    print(f"Translation '{test_word}' ({from_code} -> {to_code}): {result}")
                else:
                    print(f"No translation FOUND path from {from_code} to {to_code}")
            else:
                print(f"Language pair {from_code} -> {to_code} not available (missing code {from_code} or {to_code})")
    except Exception as e:
        print(f"Argos error: {e}")

if __name__ == "__main__":
    test()
