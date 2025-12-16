import sys
try:
    import spacy
    print(f"spacy version: {spacy.__version__}")
except ImportError:
    print("spacy not installed")

try:
    import transformers
    print(f"transformers version: {transformers.__version__}")
except ImportError:
    print("transformers not installed")

try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("nltk wordnet loaded")
    print(f"Polish synsets for 'pies': {len(wn.synsets('pies', lang='pol'))}")
except Exception as e:
    print(f"nltk error: {e}")
