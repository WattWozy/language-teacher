# Local Voice Assistant

A low-latency, local voice assistant using **FastAPI**, **Faster Whisper**, **Ollama (Llama 3.2)**, and **Piper TTS**.

## Requirements

*   Python 3.10+
*   [Ollama](https://ollama.com/)
*   NVIDIA GPU (Recommended for best performance)

## Quick Start (Windows)

Simply run the startup script:
```powershell
.\start.ps1
```
This will automatically set up the Python environment, install dependencies, and launch both servers.

## Setup (Manual)

1.  **Prepare Ollama**
    Install Ollama and pull the model:
    ```bash
    ollama pull llama3.2
    ```

2.  **Download TTS Model**
    Download `en_US-amy-low.onnx` and `en_US-amy-low.onnx.json` from [Piper Voices](https://github.com/rhasspy/piper/releases) and place them in a `models/` directory.

3.  **Install Dependencies**
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate  # Windows
    # source venv/bin/activate # Linux/Mac
    
    pip install fastapi uvicorn[standard] faster-whisper piper-tts requests soundfile numpy
    
    # For NVIDIA GPU support (Windows):
    pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
    ```

## Usage

### Backend
1.  **Start the Server**
    ```bash
    cd backend
    uvicorn server:app --host 0.0.0.0 --port 8000
    ```

### Frontend
1.  **Start the Client**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```
2.  **Open App**
    Open `http://localhost:3000` in your web browser.

## Legacy Client
*   You can still open `client.html` directly in your browser as a fallback, but the Next.js frontend is recommended.
