FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY scripts/ scripts/
COPY app.py .
COPY .streamlit/ .streamlit/
COPY convert109.sh .

# Make ffmpeg/ffprobe available at bin/ path (matches existing path resolution)
RUN mkdir -p bin && \
    ln -s /usr/bin/ffmpeg bin/ffmpeg && \
    ln -s /usr/bin/ffprobe bin/ffprobe

# Create output directories
RUN mkdir -p downloads subbed

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
