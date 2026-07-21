# Document Intelligence System — API image
#
# Build the lite image (rules + tables + TF-IDF search, no GPU stack):
#   docker build -t docintel .
# Build the full ML image (LayoutLMv3/Donut/OCR/Chroma — large):
#   docker build --build-arg FULL=1 -t docintel:full .

FROM python:3.11-slim

ARG FULL=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-lite.txt ./
RUN if [ "$FULL" = "1" ]; then \
        pip install --no-cache-dir -r requirements.txt && \
        python -m spacy download en_core_web_sm ; \
    else \
        pip install --no-cache-dir -r requirements-lite.txt ; \
    fi

COPY backend ./backend
COPY frontend ./frontend
COPY evaluation ./evaluation
COPY training ./training

# Train the supervised document classifier at build time (fast, ~10s)
RUN python training/train_classifier.py

ENV DOCINTEL_DATA_DIR=/app/data
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/chroma

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
