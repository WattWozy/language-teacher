from transformers import pipeline

try:
    # Try to load a Polish POS tagging model
    # This model is specifically trained for UPOS (Universal POS) tags
    nlp = pipeline("token-classification", model="KoichiYasuoka/bert-base-polish-upos", aggregation_strategy="simple")
    print("Model loaded successfully")
    res = nlp("To jest test.")
    print(res)
except Exception as e:
    print(f"Error loading model: {e}")
