"""Run: python -m eval.evaluate [--answers] [--oracle] [--lexical-only]."""
import argparse
import asyncio
import json
import platform
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from qdrant_client import QdrantClient

from app.search.cache import digest, write_json
from app.search.factory import PROJECT_ROOT, embedding_client, structured_client
from app.search.loader import fetch_law_root, SNAPSHOT_DATE
from app.search.pipeline import RAGPipeline
from app.search.qdrant import generate_embedding, QdrantDense
from app.search.reranker import LLMReranker
from app.search.retriever import Retriever, SearchConfig, SearchHit, article_label, pack_context
from app.search.split import chunk_law, CHUNKER_VERSION
from common.config import EMBEDDING_MODEL, MODEL, SERVICE
from .dataset import audit_corpus, load_golden
from .judge import judge_answer
from .metrics import retrieval_metrics, evidence_metrics, abstention_metrics, unique


EXPERIMENTS = {
    'A_article_dense': ('article', SearchConfig(mode='dense')),
    'B_structure_dense': ('structure', SearchConfig(mode='dense')),
    'C_structure_bm25': ('structure', SearchConfig(mode='bm25')),
    'D_hybrid_rrf': ('structure', SearchConfig()),
    'E_hybrid_rerank': ('structure', SearchConfig(rerank=True)),
    'F_rerank_expand': ('structure', SearchConfig(rerank=True, expand=True)),
}


def hit_record(hit):
    return {'key': hit.key, 'article': article_label(hit.doc), 'score': hit.score,
            'method': hit.method, 'content': hit.doc.page_content,
            'evidence_ids': hit.doc.metadata.get('evidence_ids', [])}


def score_trace(question, trace):
    required, related = question['answer_articles'], question['related_articles']
    row = {}
    for stage, hits, ks in [('candidate', trace.candidates, (20, 30)),
                            ('rank', trace.ranked, (1, 3, 5, 10))]:
        labels = [article_label(h.doc) for h in hits]
        for k in ks:
            for metric, value in retrieval_metrics(labels, required, related, k).items():
                row[f'{stage}_{metric}@{k}'] = value
    labels = unique([article_label(h.doc) for h in trace.context])
    for key, value in retrieval_metrics(labels, required, related, max(1, len(labels))).items():
        row['context_' + key] = value
    row.update(evidence_metrics([e for h in trace.context for e in h.doc.metadata.get('evidence_ids', [])],
                                question.get('required_evidence')))
    row.update(context_tokens=trace.context_tokens, retrieval_ms=trace.elapsed_ms,
               candidate_chunks=len(trace.candidates), context_chunks=len(trace.context),
               context_articles=labels, missing_articles=sorted(set(required)-set(labels)))
    return row


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    valid = frame[frame.status == 'ok']
    metrics = [c for c in frame if c.startswith(('rank_', 'candidate_', 'context_', 'evidence_'))
               and c not in {'context_articles', 'context_chunks'}]
    numeric = [c for c in metrics if c not in {'context_articles'}]
    parts = []
    for name, group in frame.groupby('experiment', sort=False):
        success = valid[valid.experiment == name]
        row = {'experiment': name, 'questions': len(group), 'succeeded': len(success),
               'failed': len(group)-len(success), 'answerable_measured': int(success.answerable.sum())}
        for col in numeric:
            values = pd.to_numeric(success.get(col, pd.Series(dtype=float)), errors='coerce').dropna()
            row[col] = float(values.mean()) if len(values) else None
        if len(success):
            row['retrieval_ms_p50'] = float(success.retrieval_ms.median())
            row['retrieval_ms_p95'] = float(success.retrieval_ms.quantile(.95))
        parts.append(row)
    return pd.DataFrame(parts)


def answer_summary(rows):
    summaries = []
    for variant in dict.fromkeys(r['experiment'] for r in rows):
        group = [r for r in rows if r['experiment'] == variant]
        refusal = abstention_metrics([r['expected_answerable'] for r in group],
                                    [r.get('response', {}).get('is_answerable') for r in group])
        row = {'experiment': variant, **refusal,
               'retrieval_errors': sum(r.get('failure_stage') == 'retrieval' for r in group),
               'generation_errors': sum(r.get('status') == 'error' and r.get('failure_stage') != 'retrieval' for r in group),
               'judge_errors': sum(r.get('judge_status') == 'error' for r in group),
               'judged': sum(r.get('judge_status') == 'ok' for r in group)}
        for key in ['correctness', 'relevance', 'point_coverage', 'faithfulness', 'citation_support', 'hallucination']:
            values = [r['judge'][key] for r in group if r.get('judge', {}).get(key) is not None]
            row[key] = sum(values)/len(values) if values else None
            row[key + '_n'] = len(values)
        values = [r['citation_validity'] for r in group if r.get('citation_validity') is not None]
        row['citation_validity'] = sum(values)/len(values) if values else None
        summaries.append(row)
    return pd.DataFrame(summaries)


async def run_benchmark(golden_path=None, output_dir=None, *, lexical_only=False,
                        answers=False, oracle=False, concurrency=3, limit=None, progress=print):
    output = Path(output_dir or PROJECT_ROOT / 'eval/results/latest')
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    output.mkdir(parents=True, exist_ok=True)
    golden_path = Path(golden_path or PROJECT_ROOT / 'golden_set_v2.jsonl')
    golden = load_golden(golden_path)
    if limit:
        golden = golden[:limit]
    if lexical_only and (answers or oracle):
        raise ValueError('lexical_only 모드에서는 LLM 평가를 실행하지 않습니다.')
    root = await fetch_law_root(PROJECT_ROOT / '.cache/rag/law.xml')
    corpora = {'structure': chunk_law(root), 'article': chunk_law(root, max_chars=100000)}
    audit = audit_corpus(root, corpora['structure'], golden)
    write_json(output / 'corpus_audit.json', audit)
    if audit['id_collisions'] or audit['unknown_golden_articles'] or audit['missing_source_units']:
        raise ValueError('원문 보존/ID/정답 조문 검증 실패: corpus_audit.json을 확인하세요.')
    experiments = {k: v for k, v in EXPERIMENTS.items() if not lexical_only or k == 'C_structure_bm25'}
    manifest = {'started_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'snapshot_date': SNAPSHOT_DATE, 'mst': SERVICE.mst, 'promulgation_no': '21311',
        'golden_file': golden_path.name, 'golden_hash': digest(golden), 'question_count': len(golden),
        'xml_hash': digest(ET.tostring(root, encoding='unicode')), 'chunker_version': CHUNKER_VERSION,
        'embedding_model': EMBEDDING_MODEL, 'llm_model': MODEL, 'python': platform.python_version(),
        'experiments': {k: {'corpus': c, **asdict(cfg)} for k, (c, cfg) in experiments.items()},
        'bm25_tokenizer': 'words+Hangul-bigrams; AI normalized; BM25Plus(delta=0)',
        'evaluation_unit': 'unique article; exact evidence only if required_evidence is labeled',
        'latency_note': 'Observed wall time; content-addressed API caches may be warm. Not cold latency.',
        'llm_judge_note': 'Same configured model as generation; automated proxy, human review required.',
        'answers': answers, 'oracle': oracle, 'lexical_only': lexical_only}
    manifest['implementation_hash'] = digest({str(p.relative_to(PROJECT_ROOT)): p.read_text()
        for folder in ['app/search', 'eval'] for p in sorted((PROJECT_ROOT / folder).glob('*.py'))})
    write_json(output / 'manifest.json', manifest)
    llm = structured_client() if not lexical_only else None
    embed = embedding_client() if not lexical_only else None
    client = QdrantClient(location=':memory:')
    retrievers = {}
    try:
        for kind in sorted({c for c, _ in experiments.values()}):
            dense = None
            if not lexical_only:
                progress(f'Indexing {kind}: {len(corpora[kind])} chunks')
                await generate_embedding(corpora[kind], client=client, embeddings=embed, collection=kind)
                dense = QdrantDense(client, embed, kind, str(SERVICE.mst))
            retrievers[kind] = Retriever(corpora[kind], dense, LLMReranker(llm) if llm else None)
        rows, details, answer_rows = [], [], []
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_one(name, kind, config, question):
            async with semaphore:
                base = {'experiment': name, 'id': question['id'], 'type': question['type'],
                        'question': question['question'], 'answerable': question['answerable']}
                try:
                    trace = await retrievers[kind].search(question['question'], config)
                    row = {**base, 'status': 'ok', **score_trace(question, trace)}
                    detail = {**base, 'status': 'ok', **{stage: [hit_record(h) for h in getattr(trace, stage)]
                              for stage in ['candidates', 'ranked', 'context']}, 'diagnostics': trace.diagnostics}
                    if name == 'F_rerank_expand' and answers:
                        answer_rows.append(await evaluate_answer(name, question, trace.context, retrievers[kind], llm))
                    return row, detail
                except Exception as exc:
                    # Exception messages may contain API keys. Record type, never raw request repr.
                    if name == 'F_rerank_expand' and answers:
                        answer_rows.append({'experiment': name, 'id': question['id'], 'type': question['type'],
                            'question': question['question'], 'expected_answerable': question['answerable'],
                            'status': 'error', 'failure_stage': 'retrieval', 'error_type': type(exc).__name__})
                    return {**base, 'status': 'error', 'error_type': type(exc).__name__}, {
                        **base, 'status': 'error', 'error_type': type(exc).__name__}

        for name, (kind, config) in experiments.items():
            progress(f'{name}: evaluating {len(golden)} questions')
            tasks = [evaluate_one(name, kind, config, q) for q in golden]
            completed = 0
            for task in asyncio.as_completed(tasks):
                row, detail = await task
                rows.append(row); details.append(detail)
                completed += 1
                if completed % 5 == 0 or completed == len(golden):
                    progress(f'  {name}: {completed}/{len(golden)}; errors={sum(r["status"] == "error" for r in rows if r["experiment"] == name)}')
                write_json(output / 'retrieval_rows.json', rows)
                write_json(output / 'retrieval_details.json', details)
                write_json(output / 'answer_rows.json', answer_rows)
            summarize(rows).to_csv(output / 'retrieval_summary.csv', index=False)

        if oracle and not lexical_only:
            # Diagnostic only: this path intentionally uses gold articles, never a retrieval score.
            async def oracle_one(question):
                async with semaphore:
                    hits = [SearchHit(d, 1, 'oracle') for d in corpora['article']
                            if article_label(d) in question['answer_articles']]
                    context, _, dropped = pack_context(hits, corpora['article'], SearchConfig(final_k=50))
                    result = await evaluate_answer('oracle_gold_articles', question, context, retrievers['structure'], llm)
                    result['oracle_complete'] = not dropped
                    return result
            tasks = [oracle_one(q) for q in golden if q['answerable']]
            for task in asyncio.as_completed(tasks):
                answer_rows.append(await task)
                write_json(output / 'answer_rows.json', answer_rows)
            progress('Oracle answer diagnostics complete')
        pd.DataFrame(rows).to_csv(output / 'retrieval_rows.csv', index=False)
        summary = summarize(rows)
        summary.to_csv(output / 'retrieval_summary.csv', index=False)
        if answer_rows:
            answer_summary(answer_rows).to_csv(output / 'answer_summary.csv', index=False)
        manifest.update(finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                        retrieval_errors=sum(r['status'] == 'error' for r in rows))
        write_json(output / 'manifest.json', manifest)
        return {'rows': rows, 'summary': summary, 'answers': answer_rows, 'audit': audit, 'manifest': manifest}
    finally:
        client.close()


async def evaluate_answer(name, question, context, retriever, llm):
    record = {'experiment': name, 'id': question['id'], 'type': question['type'],
              'question': question['question'], 'expected_answerable': question['answerable']}
    started = time.perf_counter()
    try:
        response = await RAGPipeline(retriever, llm).answer_context(question['question'], context)
        known = {h.key: h.doc.page_content for h in context}
        sources = response['sources']
        record.update(status='ok', response=response, generation_ms=(time.perf_counter()-started)*1000,
                      citation_validity=sum(known.get(s['chunk_key']) == s['content'] for s in sources)/len(sources)
                                        if sources else None)
    except Exception as exc:
        return {**record, 'status': 'error', 'error_type': type(exc).__name__}
    try:
        record.update(judge_status='ok', judge=await judge_answer(llm, question, response, context))
    except Exception as exc:
        record.update(judge_status='error', judge_error_type=type(exc).__name__)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--golden', type=Path, default=PROJECT_ROOT / 'golden_set_v2.jsonl')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'eval/results/latest')
    parser.add_argument('--lexical-only', action='store_true')
    parser.add_argument('--answers', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=3)
    args = parser.parse_args()
    result = asyncio.run(run_benchmark(args.golden, args.output, lexical_only=args.lexical_only,
                                      answers=args.answers, oracle=args.oracle, limit=args.limit,
                                      concurrency=args.concurrency))
    print(result['summary'].to_string(index=False))
