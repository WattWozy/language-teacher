import os
import sys
from langdetect import detect

# Add NVIDIA library paths to PATH for Windows
if sys.platform == "win32":
    base_path = os.path.dirname(__file__)
    nvidia_paths = [
        os.path.join(base_path, "venv", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(base_path, "venv", "Lib", "site-packages", "nvidia", "cublas", "bin"),
    ]
    for p in nvidia_paths:
        if os.path.exists(p):
            os.environ["PATH"] += ";" + p

from fastapi import FastAPI, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
from piper import PiperVoice
import requests
import soundfile as sf
import numpy as np
import io
import base64
import re
import json
from pydantic import BaseModel
from typing import List, Optional, Dict
import spacy
import nltk
from nltk.corpus import wordnet as wn
import sentencepiece as spm

# Download WordNet if needed
nltk.download("wordnet", quiet=True)

app = FastAPI()

# --- NLP Models Setup ---
# Load spaCy multilingual models
LANG_MODELS = {}
try:
    # Load models if available. User should install them: python -m spacy download en_core_web_sm
    # We try to load a few common ones.
    LANG_MODELS["en"] = spacy.load("en_core_web_sm")
    LANG_MODELS["pl"] = spacy.load("pl_core_news_sm")
    print("Loaded spaCy models")
except Exception as e:
    print(f"Warning: Could not load spaCy models: {e}")

# Load SentencePiece models
SPM_MODELS = {}
# Example: SPM_MODELS["en"] = spm.SentencePieceProcessor(model_file="models/spm_en.model")

# ---------- Helper functions for NLP ----------

def get_meanings(lemma: str):
    synsets = wn.synsets(lemma)
    if not synsets:
        return []
    return list({s.definition() for s in synsets})

def get_synonyms(lemma: str):
    synsets = wn.synsets(lemma)
    out = set()
    for s in synsets:
        for l in s.lemmas():
            out.add(l.name())
    return list(out)

def get_antonyms(lemma: str):
    synsets = wn.synsets(lemma)
    out = set()
    for s in synsets:
        for l in s.lemmas():
            if l.antonyms():
                out.add(l.antonyms()[0].name())
    return list(out)

def get_subwords(lang: str, word: str):
    sp = SPM_MODELS.get(lang)
    if not sp:
        return []
    return sp.encode(word, out_type=str)

# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for Structured Word Management ---

class WordData(BaseModel):
    translation: str
    definition: Optional[str] = None
    tags: List[str] = []
    forms: List[str] = []
    metadata: Dict = {}

class WordInput(BaseModel):
    word: str
    part_of_speech: str  # Syntactical classification (noun, verb, etc.)
    semantic_category: str = "general" # Lexical classification (food, emotion, etc.)
    data: WordData

class ClassifyRequest(BaseModel):
    word: str
    context: Optional[str] = ""

def save_word_to_file(filename: str, word_input: WordInput):
    file_path = os.path.join("roots", filename)
    if not file_path.endswith(".json"):
         file_path += ".json"
    
    os.makedirs("roots", exist_ok=True)
    
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    data = json.loads(content)
        except Exception:
            pass

    pos = word_input.part_of_speech.lower()
    cat = word_input.semantic_category.lower()
    word = word_input.word.lower()

    if pos not in data:
        data[pos] = {}
    if cat not in data[pos]:
        data[pos][cat] = {}
    
    data[pos][cat][word] = word_input.data.dict()

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return {"path": f"{pos}/{cat}/{word}", "data": word_input.data}

# --------------------------------------------------

# Load Whisper
# Try using CUDA if available, otherwise CPU
try:
    whisper = WhisperModel("small", device="cuda", compute_type="float16")
    print("Using CUDA for Whisper")
except Exception as e:
    print(f"CUDA not available for Whisper, falling back to CPU: {e}")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")

# Load Piper
voices = {}
try:
    voices["en"] = PiperVoice.load("models/en_US-amy-low.onnx")
    print("Loaded English Voice")
    voices["pl"] = PiperVoice.load("models/pl_PL-gosia-medium.onnx")
    print("Loaded Polish Voice")
    voices["no"] = PiperVoice.load("models/no_NO-talesyntese-medium.onnx")
    print("Loaded Norwegian Voice")
    voices["uk"] = PiperVoice.load("models/uk_UA-lada-x_low.onnx")
    print("Loaded Ukrainian Voice")
    voices["it"] = PiperVoice.load("models/it_IT-riccardo-x_low.onnx")
    print("Loaded Italian Voice")
except Exception as e:
    print(f"Error loading voices: {e}")

# LLM endpoint (Ollama local)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Model Configuration
CHAT_MODEL = "qwen2.5:7b"       # Smart, conversational (slower)
CLASSIFY_MODEL = "qwen2.5:0.5b" # Fast, structured tasks (faster)

# Audio Configuration
# Piper models usually output 22050Hz. 
# Lowering this value (e.g. to 18000-20000) will make the voice sound "deeper" and "slower".
PLAYBACK_SAMPLE_RATE = 20000 # Lower sample rate = Deeper/Darker pitch

def synthesize_audio(text, voice):
    # Note: The python piper-tts library version we are using does not support 'length_scale' in synthesize()
    # We rely on PLAYBACK_SAMPLE_RATE to achieve the slower/darker effect.
    chunks = list(voice.synthesize(text))
    if chunks:
        return np.concatenate([chunk.audio_int16_array for chunk in chunks])
    return np.array([], dtype=np.int16)

@app.post("/stt")
async def stt(audio: UploadFile):
    audio_bytes = await audio.read()
    def transcribe(data):
        segments, _ = whisper.transcribe(io.BytesIO(data))
        return " ".join([seg.text for seg in segments])
    
    text = await run_in_threadpool(transcribe, audio_bytes)
    return {"text": text}


@app.post("/chat")
async def chat(payload: dict):
    prompt = payload["prompt"]

    r = requests.post(OLLAMA_URL, json={"model": CHAT_MODEL, "prompt": prompt}, stream=True)
    out = ""
@app.post("/tts")
async def tts(payload: dict):
    text = payload["text"]
    # Default to English for simple endpoint
    pcm = await run_in_threadpool(synthesize_audio, text, voices["en"])

    buf = io.BytesIO()
    sf.write(buf, pcm, PLAYBACK_SAMPLE_RATE, format="WAV")
    return buf.getvalue()

@app.get("/roots/{filename}")
async def get_root_file(filename: str):
    file_path = os.path.join("roots", filename)
    if not file_path.endswith(".json"):
         file_path += ".json"
         
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                return {}
            return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roots/{filename}")
async def update_root_file(filename: str, payload: dict):
    file_path = os.path.join("roots", filename)
    if not file_path.endswith(".json"):
         file_path += ".json"
    
    os.makedirs("roots", exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        return {"status": "success", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roots/{filename}/word")
async def add_word(filename: str, word_input: WordInput):
    """
    Adds or updates a word in the structured JSON file.
    Structure: Part of Speech -> Semantic Category -> Word -> Data
    """
    try:
        result = save_word_to_file(filename, word_input)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roots/{filename}/classify")
async def classify_and_add_word(filename: str, req: ClassifyRequest):
    # Mechanical Analysis ONLY (No LLM)
    
    # 1. Determine Language
    # We try to map the filename (e.g. 'pl', 'en') to a language code
    lang_code = filename.lower()
    if lang_code not in LANG_MODELS:
        # Fallback to English if specific model not loaded, or handle error
        # For now, we'll try to use English model if available as a fallback for structure,
        # but ideally we need the specific language model.
        lang_code = "en" 
    
    nlp = LANG_MODELS.get(lang_code)
    if not nlp:
        # If absolutely no model, we can't do much mechanically
        raise HTTPException(status_code=500, detail=f"No NLP model available for language '{lang_code}'")

    # 2. Run Analysis
    doc = nlp(req.word)
    if len(doc) == 0:
         raise HTTPException(status_code=400, detail="Empty word provided")
         
    token = doc[0]
    lemma = token.lemma_
    pos = token.pos_
    
    # 3. Extract Details
    meanings = get_meanings(lemma)
    synonyms = get_synonyms(lemma)
    antonyms = get_antonyms(lemma)
    subwords = get_subwords(lang_code, req.word)
    
    # 4. Construct WordInput
    # Since we don't have an LLM to give us a "Semantic Category" (like 'food'),
    # we will default to 'general' or try to use the POS as a sub-category.
    # We also don't have a translation engine here, so 'translation' might be empty 
    # or we use the lemma.
    
    word_data = WordData(
        translation="", # Mechanical tools don't translate without a translation model
        definition=meanings[0] if meanings else "",
        tags=synonyms[:5], # Use synonyms as tags
        forms=[token.text], # Add original form
        metadata={
            "lemma": lemma,
            "pos_full": token.tag_,
            "morphology": token.morph.to_dict(),
            "meanings": meanings,
            "synonyms": synonyms,
            "antonyms": antonyms,
            "subwords": subwords
        }
    )
    
    word_input = WordInput(
        word=req.word,
        part_of_speech=pos, # e.g. NOUN, VERB
        semantic_category="general", # Mechanical limitation: hard to categorize semantically without LLM/WordNet hypernyms
        data=word_data
    )
    
    # 5. Save
    try:
        result = save_word_to_file(filename, word_input)
        
        # Return a structure similar to what the client expects
        return {
            "status": "success", 
            "classification": {
                "part_of_speech": pos,
                "semantic_category": "general",
                "translation": "",
                "definition": word_data.definition,
                "tags": word_data.tags
            }, 
            "saved": result
        }
    except Exception as e:
        print(f"Error saving word: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.get("/roots/{filename}/search")
async def search_word(filename: str, query: str):
    """
    Searches for a word across all categories in the file.
    """
    file_path = os.path.join("roots", filename)
    if not file_path.endswith(".json"):
         file_path += ".json"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
            
        # Traverse the tree: POS -> Category -> Word
        for pos, categories in data.items():
            if isinstance(categories, dict):
                for cat, words in categories.items():
                    if isinstance(words, dict):
                        for word, details in words.items():
                            if query.lower() in word.lower():
                                results.append({
                                    "word": word,
                                    "pos": pos,
                                    "category": cat,
                                    "details": details
                                })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    chat_history = []

    while True:
        message = await ws.receive()
        
        text = ""
        detected_lang = "en"

        if "bytes" in message:
            data = message["bytes"]
            # STT (Run in thread)
            result = {"text": "", "lang": "en"}
            def transcribe(data):
                segments, info = whisper.transcribe(io.BytesIO(data))
                result["lang"] = info.language
                return " ".join([seg.text for seg in segments])
            
            text = await run_in_threadpool(transcribe, data)
            detected_lang = result["lang"]
        
        elif "text" in message:
            text = message["text"]
            try:
                detected_lang = detect(text)
            except:
                detected_lang = "en"
        
        if not text.strip():
            continue

        # Handle Context Reset
        if text.strip().lower() == "/reset":
            chat_history = []
            print("Context reset.")
            await ws.send_json({"type": "log", "role": "system", "text": "Context reset.", "lang": "en"})
            continue

        print(f"User ({detected_lang}): {text}")
        await ws.send_json({"type": "log", "role": "user", "text": text, "lang": detected_lang})
        
        # Select Voice based on detected language
        # Default to English if language not supported
        current_voice = voices.get(detected_lang, voices["en"])

        # Update History
        chat_history.append({"role": "user", "content": text})

        # Streaming Pipeline
        # 1. Stream text from LLM
        # 2. Buffer into sentences
        # 3. TTS each sentence
        # 4. Stream audio back
        
        # RECOMMENDATION: Use 'qwen2.5' for better Polyglot support (Polish, Swedish, etc.)
        # Make sure to run: `ollama pull qwen2.5`
        model_name = "qwen2.5" 
        
        # System prompt to define the persona
        system_prompt = (
            f"You are a helpful polyglot language teacher. The user is speaking {detected_lang}. "
            "Reply in the same language. Keep your answers concise and conversational (1-2 sentences). "
            "If the user makes a grammar mistake, gently correct it before answering. "
            "IMPORTANT: Do not use characters from other scripts (e.g. no Chinese characters if speaking Italian). "
            "Only use the alphabet appropriate for the target language."
        )

        # Construct messages
        messages = [{"role": "system", "content": system_prompt}] + chat_history

        payload = {
            "model": model_name, 
            "messages": messages, 
            "stream": True
        }
        
        # We need to run the request in a thread, but iterate the response
        # Since requests.post with stream=True returns an iterator, we can't easily 
        # await the whole thing in run_in_threadpool without blocking streaming.
        # Ideally we'd use an async client like httpx, but let's stick to requests for now
        # and run the generator in a way that doesn't block the event loop too much.
        # Actually, for true streaming, we should process chunks as they come.
        
        # Simple sentence splitter regex
        sentence_endings = re.compile(r'[.!?]+')
        
        current_sentence = ""
        full_response = ""
        
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
                for chunk in r.iter_lines():
                    if chunk:
                        try:
                            j = json.loads(chunk.decode()) # ollama returns valid json objects per line
                            
                            # Adjust for chat API format
                            if "message" in j and "content" in j["message"]:
                                token = j["message"]["content"]
                                current_sentence += token
                                full_response += token
                                
                                # Check if we have a full sentence
                                if sentence_endings.search(token):
                                    # We found a sentence ending in this token
                                    # It's a heuristic, but works for simple TTS streaming
                                    to_speak = current_sentence.strip()
                                    if to_speak:
                                        print(f"Speaking ({detected_lang}): {to_speak}")
                                        await ws.send_json({"type": "log", "role": "assistant", "text": to_speak, "lang": detected_lang})
                                        # Generate audio for this sentence
                                        pcm = await run_in_threadpool(synthesize_audio, to_speak, current_voice)
                                        if len(pcm) > 0:
                                            buf = io.BytesIO()
                                            sf.write(buf, pcm, PLAYBACK_SAMPLE_RATE, format="WAV")
                                            audio_bytes = buf.getvalue()
                                            await ws.send_json({"type": "audio", "data": base64.b64encode(audio_bytes).decode()})
                                    
                                    current_sentence = ""
                                
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")
                            
            # Process any remaining text
            if current_sentence.strip():
                print(f"Speaking (final): {current_sentence}")
                await ws.send_json({"type": "log", "role": "assistant", "text": current_sentence, "lang": detected_lang})
                pcm = await run_in_threadpool(synthesize_audio, current_sentence, current_voice)
                if len(pcm) > 0:
                    buf = io.BytesIO()
                    sf.write(buf, pcm, PLAYBACK_SAMPLE_RATE, format="WAV")
                    audio_bytes = buf.getvalue()
                    await ws.send_json({"type": "audio", "data": base64.b64encode(audio_bytes).decode()})
            
            # Append assistant response to history
            chat_history.append({"role": "assistant", "content": full_response})
                    
        except Exception as e:
            print(f"LLM Error: {e}")