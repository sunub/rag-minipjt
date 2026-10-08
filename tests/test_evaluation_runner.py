import unittest
from eval.metrics import abstention_metrics, evidence_metrics
from eval.evaluate import summarize, score_trace
from app.search.retriever import SearchTrace, SearchHit
from langchain_core.documents import Document


class EvaluationTests(unittest.TestCase):
    def test_errors_are_counted_and_not_averaged_as_success(self):
        rows = [dict(experiment='x', status='error', answerable=True),
                dict(experiment='x', status='ok', answerable=True, retrieval_ms=2.,
                     **{'rank_recall@5': .5})]
        result = summarize(rows).iloc[0]
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['rank_recall@5'], .5)

    def test_refusal_metrics_count_missing_separately(self):
        result = abstention_metrics([False, False, True, True], [False, True, False, None])
        self.assertEqual(result['refusal_precision'], .5)
        self.assertEqual(result['refusal_recall'], .5)
        self.assertEqual(result['missing'], 1)

    def test_missing_evidence_labels_are_not_inferred_from_articles(self):
        self.assertIsNone(evidence_metrics(['x:A2'], None)['evidence_recall'])

    def test_context_score_uses_only_packed_sources(self):
        doc = Document(page_content='근거', metadata={'chunk_key':'a', 'source_type':'article', 'article_no':'2'})
        trace = SearchTrace([SearchHit(doc, 1)], [SearchHit(doc, 1)], [], 1, 0)
        row = score_trace({'answer_articles':['제2조'], 'related_articles':[]}, trace)
        self.assertEqual(row['rank_recall@1'], 1)
        self.assertEqual(row['context_recall'], 0)


if __name__ == '__main__':
    unittest.main()
