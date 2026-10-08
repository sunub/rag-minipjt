import math
import unittest
from langchain_core.documents import Document

from app.search import retriever


def doc(key, article, content, **extra):
    return Document(page_content=content, metadata={
        'chunk_key': key, 'article_no': str(article), 'article_branch_no': None,
        'law_version': 'v1', 'source_type': 'article', 'chunk_role': 'child',
        **extra,
    })


class RetrievalTests(unittest.TestCase):
    def test_bm25_returns_only_matches_and_keeps_branch(self):
        self.assertTrue(hasattr(retriever, 'BM25Index'), 'BM25Index must be implemented')
        docs = [doc('a', 22, '국제협력'), doc('b', 22, '인공지능연구소 발기인 허가', article_branch_no='2')]
        index = retriever.BM25Index(docs)
        self.assertEqual(index.search('발기인', 5)[0].doc.metadata['chunk_key'], 'b')
        self.assertEqual(index.search('zzzz', 5), [])

    def test_rrf_counts_document_only_once_per_list(self):
        self.assertTrue(hasattr(retriever, 'rrf'), 'rrf must be implemented')
        a, b = doc('a', 1, '목적'), doc('b', 2, '정의')
        hit = retriever.SearchHit
        result = retriever.rrf([[hit(a, 9), hit(a, 8), hit(b, 7)], [hit(b, 1)]])
        self.assertEqual(result[0].doc.metadata['chunk_key'], 'b')
        self.assertEqual(len(result), 2)


class MetricTests(unittest.TestCase):
    def metrics(self):
        from eval import metrics
        return metrics

    def test_duplicate_articles_do_not_inflate_recall_or_rank(self):
        m = self.metrics()
        result = m.retrieval_metrics(['제2조', '제2조', '제34조'], ['제2조', '제34조'], [], k=2)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['all_required'], 1)
        self.assertEqual(result['ndcg'], 1)

    def test_ndcg_and_first_hit_are_hand_calculated(self):
        result = self.metrics().retrieval_metrics(['제1조', '제2조'], ['제2조'], [], k=2)
        self.assertAlmostEqual(result['ndcg'], 1 / math.log2(3))
        self.assertEqual(result['mrr'], .5)

    def test_unanswerable_retrieval_is_unmeasured_not_perfect(self):
        result = self.metrics().retrieval_metrics(['제35조'], [], ['제35조'], k=5)
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['all_required'])
        self.assertIsNone(result['ndcg'])


if __name__ == '__main__':
    unittest.main()
