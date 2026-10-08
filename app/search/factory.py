from pathlib import Path
from common.ai_model import get_embedding_model, get_llm_model
from common.config import BASE_URL, EMBEDDING_MODEL, MODEL
from .cache import CachedEmbeddings, digest
from .llm import StructuredLLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / '.cache' / 'rag'


def embedding_client(cache_dir=CACHE_DIR):
    return CachedEmbeddings(get_embedding_model(), Path(cache_dir) / 'embeddings',
                            digest({'model': EMBEDDING_MODEL, 'endpoint': BASE_URL}))


def structured_client(cache_dir=CACHE_DIR):
    return StructuredLLM(get_llm_model(max_tokens=4096), Path(cache_dir) / 'llm',
                         digest({'model': MODEL, 'endpoint': BASE_URL, 'temperature': 0, 'max_tokens': 4096}))
