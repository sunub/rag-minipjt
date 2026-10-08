"""Hybrid retrieval, explicit article routing and budgeted legal context expansion."""
import re
import time
from dataclasses import dataclass, field
import tiktoken
from langchain_core.documents import Document
from rank_bm25 import BM25Plus

ARTICLE_RE = re.compile(r'제\s*(\d+)\s*조(?:\s*의\s*(\d+))?')
_ENCODING = None


def token_count(text):
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding('cl100k_base')
    return len(_ENCODING.encode(text))


def article_label(doc):
    m = doc.metadata
    if m.get('source_type') == 'addendum':
        return f"부칙:{m.get('addendum_promulgation_no')}:{m.get('addendum_article_no') or 'body'}"
    if m.get('source_type') != 'article' or not m.get('article_no'):
        return None
    return f"제{m['article_no']}조" + (f"의{m['article_branch_no']}" if m.get('article_branch_no') else '')


def tokenize(text):
    # Korean baseline: words + Hangul bigrams tolerate particles. Not a morphological analyzer.
    text = re.sub(r'\bAI\b', '인공지능', text, flags=re.I).lower()
    words = re.findall(r'[가-힣]+|[a-z0-9]+', text)
    result = list(words)
    for word in words:
        if re.fullmatch('[가-힣]{3,}', word):
            result.extend('~' + word[i:i+2] for i in range(len(word)-1))
    result.extend('제' + a + '조' + ('의' + b if b else '') for a, b in ARTICLE_RE.findall(text))
    return result


@dataclass
class SearchHit:
    doc: Document
    score: float
    method: str = ''

    @property
    def key(self):
        return self.doc.metadata['chunk_key']


class BM25Index:
    def __init__(self, docs):
        self.docs = list(docs)
        self.tokens = [tokenize(d.page_content) for d in self.docs]
        self.model = BM25Plus(self.tokens, delta=0) if self.tokens and any(self.tokens) else None

    def search(self, question, limit=20):
        if not self.model:
            return []
        scores = self.model.get_scores(tokenize(question))
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], self.docs[i].metadata['chunk_key']))
        return [SearchHit(self.docs[i], float(scores[i]), 'bm25') for i in indices[:limit] if scores[i] > 0]


def rrf(rankings, constant=60):
    scores, docs = {}, {}
    for ranking in rankings:
        seen = set()
        for hit in ranking:
            if hit.key in seen:
                continue
            seen.add(hit.key)
            docs[hit.key] = hit.doc
            scores[hit.key] = scores.get(hit.key, 0.) + 1 / (constant + len(seen))
    return [SearchHit(docs[k], score, 'rrf') for k, score in
            sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


@dataclass(frozen=True)
class SearchConfig:
    mode: str = 'hybrid'
    candidate_k: int = 30
    branch_k: int = 20
    final_k: int = 8
    rerank: bool = False
    expand: bool = False
    exact_routing: bool = False
    token_budget: int = 6000
    max_references: int = 3

    def __post_init__(self):
        if self.mode not in {'dense', 'bm25', 'hybrid'}:
            raise ValueError('Unsupported retrieval mode')
        if min(self.candidate_k, self.branch_k, self.final_k, self.token_budget) < 1:
            raise ValueError('Search limits must be positive')


@dataclass
class SearchTrace:
    candidates: list[SearchHit]
    ranked: list[SearchHit]
    context: list[SearchHit]
    elapsed_ms: float
    context_tokens: int
    diagnostics: dict = field(default_factory=dict)


def pack_context(hits, documents, config):
    """Whole chunks only: never truncate a legal exception and claim it was supplied."""
    parents = {d.metadata['chunk_key']: d for d in documents}
    by_article = {}
    for d in documents:
        label = article_label(d)
        if label and (d.metadata.get('chunk_role') == 'parent' or d.metadata.get('unit_level') == 'article'):
            by_article[label] = d
    packed, keys, covered = [], set(), set()
    used, dropped = 0, []

    def add(hit, fallback=None):
        nonlocal used
        if hit.key in keys or hit.doc.metadata.get('parent_id') in covered:
            return
        cost = token_count(hit.doc.page_content) + 32
        if used + cost > config.token_budget:
            if fallback and fallback.key != hit.key:
                add(fallback)
            else:
                dropped.append(hit.key)
            return
        packed.append(hit)
        keys.add(hit.key)
        if hit.doc.metadata.get('chunk_role') == 'parent':
            covered.add(hit.key)
        used += cost

    for hit in hits[:config.final_k]:
        parent = parents.get(hit.doc.metadata.get('parent_id')) if config.expand else None
        add(SearchHit(parent, hit.score, 'parent') if parent else hit, fallback=hit)
    if config.expand:
        refs = []
        for hit in list(packed):
            for line in hit.doc.page_content.splitlines():
                if '「' in line or '」' in line:
                    continue  # never resolve a named foreign statute as this law
                refs.extend('제' + a + '조' + ('의' + b if b else '') for a, b in ARTICLE_RE.findall(line))
        selected_articles = {article_label(h.doc) for h in packed}
        count = 0
        for ref in dict.fromkeys(refs):
            if ref in by_article and ref not in selected_articles and count < config.max_references:
                add(SearchHit(by_article[ref], 0, 'reference'))
                selected_articles.add(ref)
                count += 1
    return packed, used, dropped


class Retriever:
    def __init__(self, documents, dense=None, reranker=None, law_version=None):
        versions = {d.metadata.get('law_version') for d in documents}
        if law_version is None and len(versions) != 1:
            raise ValueError('혼합 법령 버전에서는 law_version을 지정하세요.')
        self.version = law_version or next(iter(versions))
        self.documents = [d for d in documents if d.metadata.get('law_version') == self.version]
        self.children = [d for d in self.documents if d.metadata.get('chunk_role') == 'child'
                         and d.metadata.get('source_type') == 'article']
        self.bm25 = BM25Index(self.children)
        self.dense, self.reranker = dense, reranker

    async def search(self, question, config=None):
        config = config or SearchConfig()
        if not question.strip():
            raise ValueError('질문이 비어 있습니다.')
        started = time.perf_counter()
        lists = []
        if config.mode in {'dense', 'hybrid'}:
            if self.dense is None:
                raise ValueError('Dense 검색기가 설정되지 않았습니다.')
            lists.append(await self.dense.search(question, config.branch_k))
        if config.mode in {'bm25', 'hybrid'}:
            lists.append(self.bm25.search(question, config.branch_k))
        candidates = (rrf(lists) if config.mode == 'hybrid' else lists[0])[:config.candidate_k]
        explicit = []
        if config.exact_routing:
            # Conservative named-statute scope: a foreign named law disables pinning.
            named_laws = re.findall(r'「([^」]+)」', question)
            own_names = {'인공지능기본법', 'AI기본법', '인공지능발전과신뢰기반조성등에관한기본법'}
            foreign = any(re.sub(r'\s+', '', name) not in own_names for name in named_laws)
            targets = set() if foreign else {'제' + a + '조' + ('의' + b if b else '')
                                            for a, b in ARTICLE_RE.findall(question)}
            explicit = [SearchHit(d, 1., 'exact') for d in self.children if article_label(d) in targets]
            seen = {h.key for h in explicit}
            candidates = (explicit + [h for h in candidates if h.key not in seen])[:config.candidate_k]
        ranked = list(candidates)
        if config.rerank:
            if self.reranker is None:
                raise ValueError('Reranker가 설정되지 않았습니다.')
            ranked = await self.reranker.rerank(question, candidates)
        if explicit:
            seen = {h.key for h in explicit}
            ranked = explicit + [h for h in ranked if h.key not in seen]
        context, tokens, dropped = pack_context(ranked, self.documents, config)
        return SearchTrace(candidates, ranked, context, (time.perf_counter()-started)*1000,
                           tokens, {'dropped_for_budget': dropped, 'law_version': self.version})
