"""Application configuration.

All settings can be overridden with environment variables prefixed with
``DOCINTEL_`` (e.g. ``DOCINTEL_DATA_DIR=/srv/data``). Heavy ML extractors are
enabled by default but degrade gracefully when their dependencies are not
installed — see backend/app/pipeline/extractors/.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="DOCINTEL_", env_file=".env", extra="ignore"
        )

        data_dir: Path = Path("./data")
        database_url: str = "sqlite:///./data/docintel.db"

        # extractor toggles
        enable_layoutlm: bool = True
        enable_donut: bool = True
        enable_ocr: bool = True

        # model checkpoints
        layoutlm_model: str = "nielsr/layoutlmv3-finetuned-funsd"
        donut_model: str = "naver-clova-ix/donut-base-finetuned-docvqa"

        # RAG
        embedding_model: str = "all-MiniLM-L6-v2"
        chunk_size: int = 900
        chunk_overlap: int = 150
        llm_model: str = "claude-opus-4-8"

        max_upload_mb: int = 25

except ImportError:  # pragma: no cover - lite fallback without pydantic-settings
    class Settings:  # type: ignore[no-redef]
        def __init__(self) -> None:
            env = os.environ.get
            self.data_dir = Path(env("DOCINTEL_DATA_DIR", "./data"))
            self.database_url = env(
                "DOCINTEL_DATABASE_URL", "sqlite:///./data/docintel.db"
            )
            self.enable_layoutlm = env("DOCINTEL_ENABLE_LAYOUTLM", "true") == "true"
            self.enable_donut = env("DOCINTEL_ENABLE_DONUT", "true") == "true"
            self.enable_ocr = env("DOCINTEL_ENABLE_OCR", "true") == "true"
            self.layoutlm_model = env(
                "DOCINTEL_LAYOUTLM_MODEL", "nielsr/layoutlmv3-finetuned-funsd"
            )
            self.donut_model = env(
                "DOCINTEL_DONUT_MODEL", "naver-clova-ix/donut-base-finetuned-docvqa"
            )
            self.embedding_model = env("DOCINTEL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.chunk_size = int(env("DOCINTEL_CHUNK_SIZE", "900"))
            self.chunk_overlap = int(env("DOCINTEL_CHUNK_OVERLAP", "150"))
            self.llm_model = env("DOCINTEL_LLM_MODEL", "claude-opus-4-8")
            self.max_upload_mb = int(env("DOCINTEL_MAX_UPLOAD_MB", "25"))


settings = Settings()

# Ensure runtime directories exist
for sub in ("uploads", "processed", "chroma"):
    (settings.data_dir / sub).mkdir(parents=True, exist_ok=True)
