import os
from dataclasses import dataclass
from enum import StrEnum

from dotenv import load_dotenv

load_dotenv(override=True)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
TEMPERATURE = os.getenv("LLM_TEMPERATURE", 2)
MAX_TOKENS = os.getenv("LLM_MAX_TOKENS", 2086)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "law_articles"


class ServiceTarget(StrEnum):
    LAW = "law"


@dataclass(frozen=True)
class ServiceSettings:
    base_url: str | None = os.getenv("SERVICE_BASE_URL")
    oc_key: str | None = os.getenv("OC_KEY")
    target: ServiceTarget = ServiceTarget.LAW
    mst: int = 282791  # 법령일련번호(MST)


SERVICE = ServiceSettings()
