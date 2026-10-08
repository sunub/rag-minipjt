import json
import re
from collections import Counter
from pathlib import Path

from app.search.cache import digest
from app.search.retriever import article_label


def load_golden(path):
    rows, ids = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ['id', 'type', 'question', 'answerable', 'answer_articles', 'related_articles', 'answer_points']
        if any(key not in row for key in required):
            raise ValueError(f'골든셋 {line_no}행 필수 필드 누락')
        if row['id'] in ids or not isinstance(row['answerable'], bool):
            raise ValueError(f'골든셋 {line_no}행 ID 중복 또는 answerable 오류')
        if not row['question'].strip() or (row['answerable'] and not row['answer_articles']):
            raise ValueError(f'골든셋 {line_no}행 질문/정답 조문 확인 필요')
        if not row['answerable'] and row['answer_articles']:
            raise ValueError('답변 불가능 문항의 정답 조문은 빈 배열이어야 합니다.')
        ids.add(row['id'])
        rows.append(row)
    if not rows:
        raise ValueError('골든셋이 비어 있습니다.')
    return rows


def audit_corpus(root, documents, golden):
    normalize = lambda s: re.sub(r'\s+', '', s)
    children = [d for d in documents if d.metadata.get('chunk_role') == 'child'
                and d.metadata.get('source_type') == 'article']
    texts = [normalize(d.page_content) for d in children]
    units = [e for e in root.findall('./조문/조문단위') if e.findtext('조문여부') == '조문']
    source_texts = [(e.tag, (e.text or '').strip()) for u in units for e in u.iter()
                    if e.tag in {'조문내용', '항내용', '호내용', '목내용'} and (e.text or '').strip()]
    missing = [{'tag': tag, 'text': content} for tag, content in source_texts
               if not any(normalize(content) in text for text in texts)]
    labels = {article_label(d) for d in documents}
    unknown = sorted({a for q in golden for a in q['answer_articles'] + q['related_articles']} - labels)
    keys = [d.metadata['chunk_key'] for d in documents]
    return {'xml_articles': len(units), 'chunks': len(documents), 'searchable_chunks': len(children),
            'source_units': len(source_texts), 'missing_source_units': missing,
            'source_unit_retention': 1-len(missing)/len(source_texts) if source_texts else None,
            'id_collisions': len(keys)-len(set(keys)), 'unknown_golden_articles': unknown,
            'by_role': dict(Counter(d.metadata.get('chunk_role') for d in documents)),
            'golden_count': len(golden), 'golden_hash': digest(golden),
            'evidence_labeled_questions': sum(bool(q.get('required_evidence')) for q in golden)}
