FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install git if not present
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Remove torchvision/torchaudio (not needed for text)
RUN pip uninstall -y torchvision torchaudio || true

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Download google/translategemma-4b-it
COPY <<'DOWNLOAD_SCRIPT' /tmp/download_model.py
import sys
try:
    from huggingface_hub import snapshot_download
    print("Starting download of google/translategemma-4b-it...", flush=True)
    snapshot_download(
        repo_id="google/translategemma-4b-it",
        local_dir="/models/hf/translategemma",
        local_dir_use_symlinks=False,
        token="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"
    )
    print("Download complete!", flush=True)
except Exception as e:
    print(f"DOWNLOAD FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
DOWNLOAD_SCRIPT
RUN python3 /tmp/download_model.py

ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV VLLM_NO_USAGE_STATS=1

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
