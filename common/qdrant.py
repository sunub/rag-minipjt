from qdrant_client import QdrantClient

from common.config import QDRANT_URL

def get_qdrant_client():

    return QdrantClient(
        url=QDRANT_URL
    )