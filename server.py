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

from fastapi import FastAPI, UploadFile, WebSocket
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

app = FastAPI()

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

    r = requests.post(OLLAMA_URL, json={"model": "qwen2.5", "prompt": prompt}, stream=True)
    out = ""
@app.post("/tts")
async def tts(payload: dict):
    text = payload["text"]
    # Default to English for simple endpoint
    pcm = await run_in_threadpool(synthesize_audio, text, voices["en"])

    buf = io.BytesIO()
    sf.write(buf, pcm, PLAYBACK_SAMPLE_RATE, format="WAV")
    return buf.getvalue()

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
            continue

        print(f"User ({detected_lang}): {text}")
        
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
                                        # Generate audio for this sentence
                                        pcm = await run_in_threadpool(synthesize_audio, to_speak, current_voice)
                                        if len(pcm) > 0:
                                            buf = io.BytesIO()
                                            sf.write(buf, pcm, PLAYBACK_SAMPLE_RATE, format="WAV")
                                            audio_bytes = buf.getvalue()
                                            await ws.send_text(base64.b64encode(audio_bytes).decode())
                                    
                                    current_sentence = ""
                                
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")
                            
            # Process any remaining text
            if current_sentence.strip():
                print(f"Speaking (final): {current_sentence}")
                pcm = await run_in_threadpool(synthesize_audio, current_sentence, current_voice)
                if len(pcm) > 0:
                    buf = io.BytesIO()
                    sf.write(buf, pcm, PLAYBACK_SAMPLE_RATE, format="WAV")
                    audio_bytes = buf.getvalue()
                    await ws.send_text(base64.b64encode(audio_bytes).decode())
            
            # Append assistant response to history
            chat_history.append({"role": "assistant", "content": full_response})
                    
        except Exception as e:
            print(f"LLM Error: {e}")