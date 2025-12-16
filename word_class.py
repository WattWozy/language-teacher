from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import nltk
from nltk.corpus import wordnet as wn
import sentencepiece as spm
import os

# Download WordNet if needed
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

app = FastAPI()

# Map ISO 639-1 to WordNet language codes
WN_LANG_MAP = {
    "en": "eng",
    "es": "spa",
    "it": "ita",
    "nb": "nob",
    "pl": "pol"
}

# Load spaCy multilingual models
LANG_MODELS = {
    "en": spacy.load("en_core_web_sm"),
    "es": spacy.load("es_core_news_sm"),
    "it": spacy.load("it_core_news_sm"),
    "nb": spacy.load("nb_core_news_sm"), 
    "pl": spacy.load("pl_core_news_sm")
}

# Load SentencePiece multilingual models
SPM_MODELS = {
    "en": spm.SentencePieceProcessor(model_file="models/spm_en.model"),
    "es": spm.SentencePieceProcessor(model_file="models/spm_es.model"),
    "it": spm.SentencePieceProcessor(model_file="models/spm_it.model"),
    "nb": spm.SentencePieceProcessor(model_file="models/spm_nb.model"), 
    "pl": spm.SentencePieceProcessor(model_file="models/spm_pl.model") 
}


class WordRequest(BaseModel):
    word: str
    lang: str = "en"    # "en", "es", extendable


# ---------- Helper functions ----------

def get_meanings(lemma: str, lang: str):
    wn_lang = WN_LANG_MAP.get(lang, "eng")
    try:
        synsets = wn.synsets(lemma, lang=wn_lang)
    except:
        return []
        
    if not synsets:
        return []
    return list({s.definition() for s in synsets})


def get_synonyms(lemma: str, lang: str):
    wn_lang = WN_LANG_MAP.get(lang, "eng")
    try:
        synsets = wn.synsets(lemma, lang=wn_lang)
    except:
        return []
        
    out = set()
    for s in synsets:
        for l in s.lemmas(lang=wn_lang):
            out.add(l.name())
    return list(out)


def get_antonyms(lemma: str, lang: str):
    wn_lang = WN_LANG_MAP.get(lang, "eng")
    try:
        synsets = wn.synsets(lemma, lang=wn_lang)
    except:
        return []
        
    out = set()
    for s in synsets:
        for l in s.lemmas(lang=wn_lang):
            if l.antonyms():
                out.add(l.antonyms()[0].name())
    return list(out)


def get_subwords(lang: str, word: str):
    sp = SPM_MODELS.get(lang)
    if not sp:
        return []
    return sp.encode(word, out_type=str)


# ---------- Main endpoint ----------

@app.post("/analyze")
def analyze(req: WordRequest):
    lang = req.lang.lower()

    if lang not in LANG_MODELS:
        return {"error": f"Language '{lang}' not supported."}

    nlp = LANG_MODELS[lang]
    doc = nlp(req.word)
    token = doc[0]

    lemma = token.lemma_

    return {
        "input": req.word,
        "lang": lang,
        "analysis": {
            "lemma": lemma,
            "pos": token.pos_,  # Added UPOS
            "pos_full": token.tag_ if token.tag_ else "",
            "morphology": token.morph.to_dict(),
            "meanings": get_meanings(lemma, lang),
            "synonyms": get_synonyms(lemma, lang),
            "antonyms": get_antonyms(lemma, lang),
            "subwords": get_subwords(lang, req.word)
        }
    }
