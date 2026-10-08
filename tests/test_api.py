import unittest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.main import app
from app.search.qdrant import generate_embedding
from app.search.pipeline import GroundedAnswer
from tests.test_integration import Embeddings


class FakeLLM:
    async def invoke(self, schema, system, payload):
        if schema == GroundedAnswer:
            return schema(answer='법의 목적은 권익 보호입니다. [S1]', is_answerable=True,
                          cited_source_ids=['S1'])
        return schema(scores=[{'index': i, 'score': 4} for i in range(len(payload['candidates']))])


class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_blank_question_returns_422_without_network(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            self.assertEqual((await client.post('/ask', json={'question': '  '})).status_code, 422)

    async def test_ask_returns_real_stored_source(self):
        from common.config import SERVICE
        db = QdrantClient(':memory:')
        text = '제1조 목적: 국민의 권익을 보호한다.'
        doc = Document(page_content=text, metadata={'chunk_key':'a', 'chunk_role':'child',
            'source_type':'article', 'law_version':str(SERVICE.mst), 'article_no':'1',
            'unit_level':'article'})
        await generate_embedding([doc], client=db, embeddings=Embeddings())
        with patch('app.main.get_qdrant_client', return_value=db), \
             patch('app.main.embedding_client', return_value=Embeddings()), \
             patch('app.main.structured_client', return_value=FakeLLM()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/ask', json={'question':'법의 목적은?'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['sources'][0]['content'], text)
        self.assertEqual(response.json()['sources'][0]['article'], '제1조')


if __name__ == '__main__':
    unittest.main()
