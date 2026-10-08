"""Qdrant persistence with model/corpus fingerprints and non-destructive schema checks."""
import uuid
from langchain_core.documents import Document
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
    PayloadSchemaType, PointStruct, PointIdsList, VectorParams)
from common.ai_model import get_embedding_model
from common.config import COLLECTION_NAME, EMBEDDING_MODEL, BASE_URL
from common.qdrant import get_qdrant_client
from .cache import digest

BATCH_SIZE = 64
PAYLOAD_INDEXES = ['law_id', 'law_version', 'source_type', 'chunk_role', 'chunk_key',
                   'parent_id', 'article_no', 'article_branch_no', 'evidence_ids', 'embedding_identity']


def embedding_identity(embeddings, model_id=EMBEDDING_MODEL):
    return getattr(embeddings, 'identity', None) or digest({
        'model': getattr(embeddings, 'model', model_id),
        'endpoint': getattr(embeddings, 'openai_api_base', BASE_URL)})


def make_point_id(doc):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc.metadata['chunk_key']))


def metadata_filter(**values):
    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in values.items() if v is not None])


def load_documents(client, collection=COLLECTION_NAME):
    docs, offset = [], None
    if not client.collection_exists(collection):
        return docs
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for point in points:
            payload = dict(point.payload or {})
            content = payload.pop('page_content', '')
            docs.append(Document(page_content=content, metadata=payload))
        if offset is None:
            return sorted(docs, key=lambda d: d.metadata['chunk_key'])


async def generate_embedding(documents, *, client=None, embeddings=None,
                             collection=COLLECTION_NAME, model_id=EMBEDDING_MODEL):
    if not documents:
        raise ValueError('빈 문서는 인덱싱할 수 없습니다.')
    client = client if client is not None else get_qdrant_client()
    embeddings = embeddings if embeddings is not None else get_embedding_model()
    ids = [make_point_id(d) for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError('chunk_key가 중복되어 Point ID가 충돌합니다.')
    identity = embedding_identity(embeddings, model_id)
    fingerprints = {d.metadata['chunk_key']: digest({'text': d.page_content,
                    'metadata': d.metadata, 'embedding_identity': identity}) for d in documents}
    old_docs = load_documents(client, collection)
    if old_docs and {d.metadata['chunk_key']: d.metadata.get('index_fingerprint')
                     for d in old_docs} == fingerprints:
        return 0
    vectors = await embeddings.aembed_documents([d.page_content for d in documents])
    if len(vectors) != len(documents) or not vectors[0]:
        raise ValueError('임베딩 결과 개수/차원이 잘못되었습니다.')
    dimensions = len(vectors[0])
    if any(len(v) != dimensions for v in vectors):
        raise ValueError('임베딩 차원이 일치하지 않습니다.')
    if client.collection_exists(collection):
        config = client.get_collection(collection).config.params.vectors
        if not isinstance(config, VectorParams) or config.size != dimensions or config.distance != Distance.COSINE:
            raise ValueError('기존 컬렉션 벡터 설정이 다릅니다. 새 collection 이름을 사용하세요.')
    else:
        client.create_collection(collection, vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE))
    if type(client._client).__name__ != 'QdrantLocal':
        for field in PAYLOAD_INDEXES:
            client.create_payload_index(collection, field, PayloadSchemaType.KEYWORD, wait=True)
    for start in range(0, len(documents), BATCH_SIZE):
        batch = [PointStruct(id=ids[i], vector=vectors[i], payload={
            **documents[i].metadata, 'page_content': documents[i].page_content,
            'embedding_model': model_id,
            'embedding_identity': identity,
            'index_fingerprint': fingerprints[documents[i].metadata['chunk_key']],
        }) for i in range(start, min(start + BATCH_SIZE, len(documents)))]
        client.upsert(collection, points=batch, wait=True)
    stale = [make_point_id(d) for d in old_docs if d.metadata['chunk_key'] not in fingerprints]
    if stale:
        client.delete(collection, points_selector=PointIdsList(points=stale), wait=True)
    return len(documents)


class QdrantDense:
    def __init__(self, client, embeddings, collection=COLLECTION_NAME, law_version=None):
        self.client, self.embeddings = client, embeddings
        self.collection, self.law_version = collection, law_version

    async def search(self, question, limit):
        from .retriever import SearchHit
        scope = metadata_filter(chunk_role='child', law_version=self.law_version, source_type='article')
        incompatible = Filter(must=scope.must, must_not=[FieldCondition(
            key='embedding_identity', match=MatchValue(value=embedding_identity(self.embeddings)))])
        if self.client.count(self.collection, count_filter=incompatible, exact=True).count:
            raise ValueError('저장 벡터와 질의 임베딩 모델/endpoint가 다릅니다. 재인덱싱하세요.')
        vector = await self.embeddings.aembed_query(question)
        response = self.client.query_points(self.collection, query=vector, limit=limit,
            query_filter=scope, with_payload=True)
        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            content = payload.pop('page_content')
            hits.append(SearchHit(Document(page_content=content, metadata=payload),
                                  float(point.score), 'dense'))
        return hits
