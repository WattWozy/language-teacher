import os
import sys

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
#voice = PiperVoice.load("models/en_US-amy.onnx")
voice = PiperVoice.load("models/en_US-amy-low.onnx")

# LLM endpoint (Ollama local)
OLLAMA_URL = "http://localhost:11434/api/generate"

def synthesize_audio(text):
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

    r = requests.post(OLLAMA_URL, json={"model": "phi3", "prompt": prompt}, stream=True)
    out = ""
@app.post("/tts")
async def tts(payload: dict):
    text = payload["text"]
    pcm = await run_in_threadpool(synthesize_audio, text)

    buf = io.BytesIO()
    sf.write(buf, pcm, 22050, format="WAV")
    return buf.getvalue()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_bytes()  # receive raw audio
        
        # STT (Run in thread)
        def transcribe(data):
            segments, _ = whisper.transcribe(io.BytesIO(data))
            return " ".join([seg.text for seg in segments])
        
        text = await run_in_threadpool(transcribe, data)
        if not text.strip():
            continue

        print(f"User: {text}")

        # Streaming Pipeline
        # 1. Stream text from LLM
        # 2. Buffer into sentences
        # 3. TTS each sentence
        # 4. Stream audio back
        # Use a smaller model for speed if possible, e.g., "qwen2.5:3b" or "phi3.5"
        # Ensure you have pulled the model: `ollama pull llama3.2`
        model_name = "llama3.2" 
        
        payload = {"model": model_name, "prompt": text, "stream": True}
        payload = {"model": model_name, "prompt": text, "stream": True}
        
        # We need to run the request in a thread, but iterate the response
        # Since requests.post with stream=True returns an iterator, we can't easily 
        # await the whole thing in run_in_threadpool without blocking streaming.
        # Ideally we'd use an async client like httpx, but let's stick to requests for now
        # and run the generator in a way that doesn't block the event loop too much.
        # Actually, for true streaming, we should process chunks as they come.
        
        # Simple sentence splitter regex
        sentence_endings = re.compile(r'[.!?]+')
        
        current_sentence = ""
        
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
                for chunk in r.iter_lines():
                    if chunk:
                        try:
                            j = json.loads(chunk.decode()) # ollama returns valid json objects per line
                            token = j.get("response", "")
                            
                            current_sentence += token
                            
                            # Check if we have a full sentence
                            if sentence_endings.search(token):
                                # We found a sentence ending in this token
                                # It's a heuristic, but works for simple TTS streaming
                                to_speak = current_sentence.strip()
                                if to_speak:
                                    print(f"Speaking: {to_speak}")
                                    # Generate audio for this sentence
                                    pcm = await run_in_threadpool(synthesize_audio, to_speak)
                                    if len(pcm) > 0:
                                        buf = io.BytesIO()
                                        sf.write(buf, pcm, 22050, format="WAV")
                                        audio_bytes = buf.getvalue()
                                        await ws.send_text(base64.b64encode(audio_bytes).decode())
                                
                                current_sentence = ""
                                
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")
                            
            # Process any remaining text
            if current_sentence.strip():
                print(f"Speaking (final): {current_sentence}")
                pcm = await run_in_threadpool(synthesize_audio, current_sentence)
                if len(pcm) > 0:
                    buf = io.BytesIO()
                    sf.write(buf, pcm, 22050, format="WAV")
                    audio_bytes = buf.getvalue()
                    await ws.send_text(base64.b64encode(audio_bytes).decode())
                    
        except Exception as e:
            print(f"LLM Error: {e}")