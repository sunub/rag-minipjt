import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.search.qdrant import generate_embedding, load_documents, QdrantDense
from app.search.retriever import Retriever, SearchConfig, article_label, SearchHit, pack_context
from app.search.split import chunk_law
from app.search.pipeline import validate_answer
from eval.dataset import load_golden


class Embeddings:
    async def aembed_documents(self, texts):
        return [[1., 0.] for _ in texts]

    async def aembed_query(self, text):
        return [1., 0.]


def make_doc(key, version='v1', **kwargs):
    return Document(page_content='제22조의2 인공지능연구소', metadata={
        'chunk_key': key, 'chunk_role': 'child', 'source_type': 'article',
        'law_version': version, 'article_no': '22', 'article_branch_no': '2', **kwargs})


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_reuses_identical_data_but_updates_changed_metadata(self):
        client = QdrantClient(':memory:')
        try:
            docs = [make_doc('a')]
            self.assertEqual(await generate_embedding(docs, client=client, embeddings=Embeddings()), 1)
            self.assertEqual(await generate_embedding(docs, client=client, embeddings=Embeddings()), 0)
            docs[0].metadata['article_title'] = '변경된 제목'
            self.assertEqual(await generate_embedding(docs, client=client, embeddings=Embeddings()), 1)
            self.assertEqual(load_documents(client)[0].metadata['article_title'], '변경된 제목')
        finally:
            client.close()

    async def test_dense_excludes_parent_and_wrong_version(self):
        client = QdrantClient(':memory:')
        try:
            await generate_embedding([make_doc('child'), make_doc('parent', chunk_role='parent'),
                                      make_doc('old', version='v0')], client=client, embeddings=Embeddings())
            hits = await QdrantDense(client, Embeddings(), law_version='v1').search('연구소', 10)
            self.assertEqual([h.key for h in hits], ['child'])
        finally:
            client.close()

    async def test_exact_route_distinguishes_branch_and_keeps_related_search(self):
        docs = [make_doc('branch'), make_doc('plain', article_branch_no=None)]
        retriever = Retriever(docs)
        trace = await retriever.search('제22조의2 설명', SearchConfig(mode='bm25', exact_routing=True))
        self.assertEqual(article_label(trace.ranked[0].doc), '제22조의2')

    async def test_external_law_citation_does_not_pin_our_article(self):
        docs = [make_doc('branch')]
        retriever = Retriever(docs)
        trace = await retriever.search('「개인정보 보호법」 제22조의2 설명',
                                       SearchConfig(mode='bm25', exact_routing=True))
        self.assertTrue(all(h.method != 'exact' for h in trace.ranked))

    async def test_query_model_mismatch_is_rejected_before_search(self):
        client = QdrantClient(':memory:')
        try:
            first, second = Embeddings(), Embeddings()
            first.identity, second.identity = 'endpoint-model-A', 'endpoint-model-B'
            await generate_embedding([make_doc('a')], client=client, embeddings=first)
            with self.assertRaises(ValueError):
                await QdrantDense(client, second, law_version='v1').search('연구소', 10)
        finally:
            client.close()


class ContentTests(unittest.TestCase):
    def test_inline_unknown_citation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_answer({'answer': '근거 [S99]', 'is_answerable': True, 'cited_source_ids': ['S1']}, {'S1': {}})

    def test_external_law_reference_is_not_expanded(self):
        a = make_doc('a', unit_level='article', article_no='1', article_branch_no=None)
        a.page_content = '제1조 「다른법」 제22조의2에 따른다.'
        b = make_doc('b', unit_level='article')
        packed, _, _ = pack_context([SearchHit(a, 1)], [a, b], SearchConfig(expand=True))
        self.assertEqual([h.key for h in packed], ['a'])

    def test_golden_set_has_36_answerable_and_9_unanswerable(self):
        rows = load_golden(Path(__file__).resolve().parents[1] / 'golden_set_v2.jsonl')
        self.assertEqual(sum(q['answerable'] for q in rows), 36)
        self.assertEqual(sum(not q['answerable'] for q in rows), 9)

    def test_leaf_split_does_not_claim_complete_evidence(self):
        content = '매우 긴 의무 조건 및 예외 문장 ' * 100
        root = ET.fromstring(f'<법령><기본정보><법령ID>x</법령ID></기본정보><조문><조문단위>'
            f'<조문번호>1</조문번호><조문여부>조문</조문여부><조문내용>{content}</조문내용>'
            '</조문단위></조문></법령>')
        docs = chunk_law(root, max_chars=100)
        children = [d for d in docs if d.metadata['chunk_role'] == 'child']
        self.assertTrue(children)
        self.assertTrue(all('x:A1' not in d.metadata['evidence_ids'] for d in children))


if __name__ == '__main__':
    unittest.main()
