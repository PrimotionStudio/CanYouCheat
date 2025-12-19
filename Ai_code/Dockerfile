# 1. Use a 'slim' image to reduce base bloat
FROM python:3.10-slim

# 2. Set environment variables to optimize Python/Pip behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /code

# 3. Install system dependencies (added build-essential for cmake/deepface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip and set up the user FIRST
RUN pip install --upgrade pip
RUN useradd -m -u 1000 user

# 5. Install requirements as the root user BEFORE switching
# This helps avoid permission issues with large package extractions
COPY requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

# 6. Copy application, weights, and startup script
COPY app /code/app
COPY yolov8n.pt /code/yolov8n.pt

# 7. Pre-download DeepFace models as root (before switching to user)
# This ensures models are downloaded during build, not at runtime
COPY app/startup.py /code/startup.py
RUN python /code/startup.py

# 8. Switch to non-root user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
