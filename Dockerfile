# ==============================
# 1. Base image with Python
# ==============================
FROM python:3.11-slim

# ==============================
# 2. Set environment variables
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# ==============================
# 3. Install system dependencies
# ==============================
# - poppler-utils → required for pdf2image
# - ffmpeg & libsm6 & libxext6 → needed for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 4. Copy requirements and install Python deps
# ==============================
WORKDIR /app
COPY requirements.txt .

# Install PaddleOCR dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==============================
# 5. Copy application code
# ==============================
COPY . .

# ==============================
# 6. Expose FastAPI port
# ==============================
EXPOSE 8000

# ==============================
# 7. Run FastAPI with uvicorn
# ==============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]