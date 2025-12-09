import requests
import os

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

files = {
    "pl_PL-gosia-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx",
    "pl_PL-gosia-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json",
    "no_NO-talesyntese-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx",
    "no_NO-talesyntese-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx.json",
    "uk_UA-lada-x_low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx",
    "uk_UA-lada-x_low.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx.json"
}

for filename, url in files.items():
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")
