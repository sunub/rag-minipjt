import unittest
from langchain_core.documents import Document
from app.search import pipeline, reranker
from app.search.retriever import SearchHit, SearchConfig, pack_context


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_citation_fails_instead_of_becoming_a_source(self):
        self.assertTrue(hasattr(pipeline, 'validate_answer'), 'validate_answer required')
        answer = {'answer': '답변', 'is_answerable': True, 'cited_source_ids': ['S99']}
        with self.assertRaises(ValueError):
            pipeline.validate_answer(answer, {'S1': {}})

    async def test_answerable_response_requires_citation(self):
        self.assertTrue(hasattr(pipeline, 'validate_answer'))
        with self.assertRaises(ValueError):
            pipeline.validate_answer({'answer': '답변', 'is_answerable': True, 'cited_source_ids': []}, {})

    async def test_citations_must_be_attached_in_answer_text(self):
        with self.assertRaises(ValueError):
            pipeline.validate_answer({'answer': '출처가 붙지 않은 주장', 'is_answerable': True,
                                      'cited_source_ids': ['S1']}, {'S1': {}})

    async def test_bad_citation_can_be_regenerated_once_without_inventing_source(self):
        class InvalidThenValid:
            def __init__(self):
                self.calls = 0

            async def invoke(self, schema, system, payload):
                self.calls += 1
                sid = 'S99' if self.calls == 1 else 'S1'
                return schema(answer=f'답변 [{sid}]', is_answerable=True, cited_source_ids=[sid])
        doc = Document(page_content='법률 원문', metadata={'chunk_key':'a', 'source_type':'article', 'article_no':'1'})
        result = await pipeline.RAGPipeline(None, InvalidThenValid()).answer_context('질문', [SearchHit(doc, 1)])
        self.assertEqual(result['generation_attempts'], 2)
        self.assertEqual(result['sources'][0]['content'], '법률 원문')

    async def test_reranking_rejects_missing_and_duplicate_indices(self):
        self.assertTrue(hasattr(reranker, 'validate_scores'))
        for rows in [[{'index': 0, 'score': 1}], [{'index': 0, 'score': 1}, {'index': 0, 'score': 2}]]:
            with self.assertRaises(ValueError):
                reranker.validate_scores(rows, 2)

    def test_context_never_truncates_text_to_fit(self):
        doc = Document(page_content='조건과 예외 ' * 30, metadata={'chunk_key': 'a'})
        packed, used, dropped = pack_context([SearchHit(doc, 1)], [doc], SearchConfig(token_budget=1))
        self.assertEqual(packed, [])
        self.assertEqual(used, 0)
        self.assertEqual(dropped, ['a'])


if __name__ == '__main__':
    unittest.main()
