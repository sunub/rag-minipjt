"""Deterministic metrics. None means not measurable, never a perfect score."""
import math


def unique(values):
    return list(dict.fromkeys(v for v in values if v))


def retrieval_metrics(ranked, required, related=(), k=5):
    if k < 1:
        raise ValueError('k must be positive')
    required, related = set(required), set(related)
    top = unique(ranked)[:k]
    if not required:
        return dict(hit=None, recall=None, mrr=None, ndcg=None, all_required=None)
    found = required.intersection(top)
    grades = {a: 1 for a in related}
    grades.update({a: 2 for a in required})
    dcg = sum((2 ** grades.get(a, 0) - 1) / math.log2(i + 2) for i, a in enumerate(top))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {
        'hit': float(bool(found)), 'recall': len(found) / len(required),
        'mrr': next((1 / (i + 1) for i, a in enumerate(top) if a in required), 0.),
        'ndcg': dcg / idcg, 'all_required': float(required.issubset(top)),
    }


def evidence_metrics(retrieved, required):
    if not required:
        return {'evidence_recall': None, 'evidence_all_required': None}
    gold, found = set(required), set(retrieved)
    return {'evidence_recall': len(gold & found) / len(gold),
            'evidence_all_required': float(gold <= found)}


def abstention_metrics(expected_answerable, predicted_answerable):
    pairs = [(g, p) for g, p in zip(expected_answerable, predicted_answerable) if p is not None]
    tp = sum(not g and not p for g, p in pairs)
    fp = sum(g and not p for g, p in pairs)
    fn = sum(not g and p for g, p in pairs)
    tn = sum(g and p for g, p in pairs)
    return {
        'refusal_precision': tp / (tp + fp) if tp + fp else None,
        'refusal_recall': tp / (tp + fn) if tp + fn else None,
        'answerability_accuracy': (tp + tn) / len(pairs) if pairs else None,
        'true_refusal': tp, 'false_refusal': fp, 'missed_refusal': fn, 'true_answer': tn,
        'measured': len(pairs), 'missing': len(expected_answerable) - len(pairs),
    }
