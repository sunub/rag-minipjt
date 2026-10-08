"""Rebuild the reviewable notebook source; execution results are generated separately."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# AI 기본법 RAG 검색·답변 평가

`golden_set_v2.jsonl`의 **45문항(답변 가능 36 / 범위 밖 9)** 을 그대로 사용합니다.

- **A→B**: 조 전체 vs 구조 기반 청킹, **B/C/D**: Dense / BM25 / Hybrid,
  **D→E**: LLM 재정렬, **E→F**: 상위 문맥·참조 조문 확장.
- 현재 정답 라벨은 **조문 단위**입니다. 조문 검색 성공이 해당 항·호·목의 근거 충족을 보장하지 않습니다.
- 항·호·목 라벨이 없는 근거 Recall은 `NaN`(미측정), 범위 밖 질문의 검색 정답 지표도 `NaN`입니다.
- 답변/LLM judge 평가는 자동 보조 지표입니다. 생성과 judge가 같은 설정 모델을 사용하므로 독립적 법률 검증이 아닙니다.

먼저 `uv sync`, 커널은 프로젝트 `.venv`를 선택하세요. **기본 Run All은 저장 결과를 읽으며 API를 재호출하지 않습니다.**
재실행하려면 아래 `RUN_EVALUATION=True`로 바꾸세요. `MODE='full'`은 설정된 외부 API로 공개 법령/질문을 보내며 비용이 발생합니다.
''')
code('''from pathlib import Path
import sys, json, os
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'golden_set_v2.jsonl').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from eval.dataset import load_golden
from eval.evaluate import run_benchmark, summarize, answer_summary
from app.search.cache import digest
plt.rcParams.update({'figure.figsize': (11, 4), 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})
pd.set_option('display.max_colwidth', 100)
golden = load_golden(ROOT / 'golden_set_v2.jsonl')
golden_df = pd.DataFrame(golden)
RUN_EVALUATION = False
MODE = 'full'  # 'full': A~F + 답변/judge/oracle, 'lexical': API 없이 BM25 (XML 캐시 필요)
RESULT_DIR = ROOT / 'eval/results' / MODE
if os.getenv('RAG_RESULTS_DIR'):
    RESULT_DIR = Path(os.environ['RAG_RESULTS_DIR'])
elif not RUN_EVALUATION and not (RESULT_DIR / 'retrieval_rows.json').exists():
    RESULT_DIR = ROOT / 'eval/results/lexical'
print('Project:', ROOT)
print('Results:', RESULT_DIR)
''')
md('''## 1. 평가 기준과 데이터 점검

상위 K는 **중복 제거한 조문 순위**이며 후보 청크 개수와 구분됩니다.
`candidate_recall@30`도 실제 후보에서 얻은 고유 조문만 평가합니다.
정답 조문 관련도는 2, `related_articles`만 있는 조문은 1, 나머지는 0입니다.
전체 확보율은 질문의 `answer_articles`를 모두 확보한 질문의 비율입니다.
''')
code('''criteria = pd.DataFrame([
    ['데이터', '원문 단위 보존율 / ID 충돌 / 라벨 존재', 'XML과 실제 검색 청크 비교', '자동'],
    ['후보 검색', 'Recall@20 / @30', '답변 가능 36문항', '자동·조문'],
    ['순위', 'Hit / Recall / MRR / nDCG @1,3,5,10', '중복 제거 조문 순위', '자동·조문'],
    ['컨텍스트', 'Recall / 전체 확보율', '6000 토큰 예산 내 최종 전달 근거', '자동·조문'],
    ['세부 근거', 'evidence Recall / 전체 확보율', 'required_evidence 라벨 필요', '현재 미측정'],
    ['답변', '정답성 / 관련성 / 요점 충족 / 충실성 / 환각', '생성한 응답만', 'LLM judge'],
    ['인용', '출처 원문 일치 / 인용이 주장 지지', '인용이 있는 응답', '자동 + LLM judge'],
    ['거절', 'Precision / Recall / confusion matrix', '답변 가능·불가능 전체', '라벨 비교'],
], columns=['단계', '지표', '대상', '방식'])
display(criteria)
display(golden_df.groupby(['type', 'answerable']).size().rename('문항 수').reset_index())
assert golden_df.id.is_unique
print(f'총 {len(golden_df)}문항, 답변 가능 {golden_df.answerable.sum()}문항')
''')
md('''## 2. 실행 또는 저장 결과 불러오기

XML은 `.cache/rag/law.xml`, 임베딩과 LLM 응답은 모델·본문·프롬프트별 해시로 캐시합니다.
실험 Qdrant는 프로세스 메모리에 생성하므로 운영 `law_articles` 컬렉션을 변경하지 않습니다.
캐시가 있으면 지연시간은 실제 신규 API 호출의 지연과 다릅니다. 결과 JSON/CSV에는 오류를 별도 기록합니다.
''')
code('''if RUN_EVALUATION:
    result = await run_benchmark(
        golden_path=ROOT / 'golden_set_v2.jsonl', output_dir=RESULT_DIR,
        lexical_only=MODE == 'lexical', answers=MODE == 'full', oracle=MODE == 'full',
    )
if not (RESULT_DIR / 'retrieval_rows.json').exists():
    raise FileNotFoundError('저장된 결과가 없습니다. RUN_EVALUATION=True로 실행하세요.')
read_json = lambda name: json.loads((RESULT_DIR / name).read_text(encoding='utf-8'))
manifest = read_json('manifest.json')
assert manifest['golden_hash'] == digest(golden), '골든셋이 변경되었습니다. 평가를 다시 실행하세요.'
rows = read_json('retrieval_rows.json')
details = read_json('retrieval_details.json')
answer_rows = read_json('answer_rows.json') if (RESULT_DIR / 'answer_rows.json').exists() else []
audit = read_json('corpus_audit.json')
df = pd.DataFrame(rows)
summary = summarize(rows).set_index('experiment')
FIGURES = RESULT_DIR / 'figures'
FIGURES.mkdir(exist_ok=True)
def finish(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    plt.show()
display(pd.Series(manifest, name='실행 설정').to_frame())
if 'finished_at' not in manifest:
    display(Markdown('**주의: 실행이 완료되지 않은 부분 결과입니다.**'))
display(summary[['questions', 'succeeded', 'failed', 'answerable_measured']])
''')
code('''display(pd.DataFrame([{
    'XML 조문': audit['xml_articles'], '검색 본문 청크': audit['searchable_chunks'],
    '원문 텍스트 단위': audit['source_units'], '원문 보존율': audit['source_unit_retention'],
    'ID 충돌': audit['id_collisions'], '없는 정답 조문': len(audit['unknown_golden_articles']),
    '세부 근거 라벨 문항': audit['evidence_labeled_questions'],
}]))
if audit['missing_source_units']:
    display(pd.DataFrame(audit['missing_source_units']))
fig, ax = plt.subplots(figsize=(7, 2.8))
values = [audit['source_unit_retention'], float(audit['id_collisions'] == 0),
          float(not audit['unknown_golden_articles'])]
ax.barh(['Source unit retention', 'Unique chunk IDs', 'Golden articles exist'], values, color='#188977')
ax.set_xlim(0, 1.1)
for i, v in enumerate(values): ax.text(v + .01, i, f'{v:.1%}', va='center')
finish('01_integrity')
''')
md('''## 3. 검색 방식별 비교

답변 가능 질문만 검색 정답 지표 평균에 포함합니다. 범위 밖 질문 9개에 관련 조문이 있더라도 정답 검색 성공으로 취급하지 않습니다.
오류 문항은 평균에서 빠지므로 반드시 위의 **failed** 수도 함께 확인하세요.
''')
code('''columns = ['candidate_recall@30', 'rank_hit@5', 'rank_recall@5', 'rank_mrr@10',
           'rank_ndcg@5', 'context_recall', 'context_all_required', 'context_tokens']
display(summary.reindex(columns=columns).round(4))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cols, title in zip(axes,
    [['rank_recall@5', 'rank_ndcg@5'], ['context_recall', 'context_all_required'], ['candidate_recall@30', 'rank_mrr@10']],
    ['Ranking quality', 'Final context coverage', 'Candidate recall / first hit']):
    summary[cols].rename(index=lambda x: x.split('_')[0]).plot.bar(ax=ax, rot=0, color=['#2878b5', '#e99535'])
    ax.set_title(title); ax.set_ylim(0, 1.13); ax.legend(fontsize=7)
    ax.set_xlabel('Experiment')
finish('02_comparison')
''')
code('''fig, ax = plt.subplots(figsize=(9, 4))
for name, row in summary.iterrows():
    ks = [1, 3, 5, 10]
    ax.plot(ks, [row[f'rank_recall@{k}'] for k in ks], marker='o', label=name)
ax.set(xlabel='K (unique articles)', ylabel='Macro Recall@K', ylim=(0, 1.05), xticks=ks)
ax.legend(fontsize=8, loc='lower right')
finish('03_recall_curve')
''')
md('''## 4. 질문 유형별 품질과 실패 질문

전체 평균이 좋아도 복합·숫자벌칙·신설 조문에서 약할 수 있습니다. 아래 heatmap은 유형별 최종 컨텍스트 전체 확보율입니다.
''')
code('''TYPE_EN = {'정의':'Definition', '의무':'Obligation', '일반인표현':'Everyday',
           '숫자벌칙':'Penalty', '복합':'Multi-part', '신설':'New articles', '기타':'Other', '범위밖':'Out of scope'}
ok = df[df.status.eq('ok')].copy()
answerable = ok[ok.answerable].copy()
heat = answerable.pivot_table(index='type', columns='experiment', values='context_all_required', aggfunc='mean')
display(heat.round(3))
fig, ax = plt.subplots(figsize=(max(7, len(heat.columns)*1.5), 4))
image = ax.imshow(heat.to_numpy(dtype=float), vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
ax.set_xticks(range(len(heat.columns)), [n.split('_')[0] for n in heat.columns])
ax.set_yticks(range(len(heat.index)), [TYPE_EN.get(n, n) for n in heat.index])
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        v = heat.iloc[i,j]
        ax.text(j, i, f'{v:.0%}', ha='center', va='center', color='white' if v > .65 else 'black')
fig.colorbar(image, ax=ax, label='All required articles retrieved')
finish('04_category_heatmap')
failures = answerable[answerable.context_all_required.lt(1)]
display(failures[['experiment','id','type','question','missing_articles','context_recall']].sort_values(['experiment','id']))
errors = df[df.status.ne('ok')]
if len(errors): display(errors[['experiment','id','question','error_type']])
''')
md('''## 5. 질문별 검색 결과 탐색

`QUESTION_ID`와 `EXPERIMENT`를 바꾸면 후보 순위, 최종 전달 근거, 누락 조문을 확인할 수 있습니다.
검색 점수는 Dense/BM25/RRF/Reranker마다 척도가 달라 서로 직접 비교하지 않습니다.
''')
code('''QUESTION_ID = 'q16'
EXPERIMENT = 'F_rerank_expand' if 'F_rerank_expand' in df.experiment.values else df.experiment.iloc[0]
def inspect_question(question_id, experiment):
    question = next(q for q in golden if q['id'] == question_id)
    detail = next(d for d in details if d['id'] == question_id and d['experiment'] == experiment)
    display(Markdown(f"**{question_id}: {question['question']}**"))
    print('정답 조문:', question['answer_articles'], '| 답변 가능:', question['answerable'])
    if detail['status'] != 'ok':
        display(detail); return
    for stage in ['candidates', 'ranked', 'context']:
        print(stage)
        table = pd.DataFrame(detail[stage])
        if not table.empty:
            table.insert(0, 'rank', range(1, len(table)+1))
            table['gold_article'] = table.article.isin(question['answer_articles'])
            display(table[['rank','article','gold_article','method','score','content']].head(12))
    for record in answer_rows:
        if record['id'] == question_id and record['experiment'] == experiment:
            display(record)
inspect_question(QUESTION_ID, EXPERIMENT)
''')
md('''## 6. 동일 질문 기준 개선량과 비용

가능한 경우 D(Hybrid)를 기준으로 E/F의 질문별 차이를 비교합니다. 부트스트랩 신뢰구간은
동일 문항을 묶어 2,000번 재표집한 기술적 요약입니다. 45문항은 작은 평가셋이고,
이 결과로 설정을 고르면 더 이상 독립된 최종 테스트셋이 아닙니다.
''')
code('''baseline = 'D_hybrid_rrf' if 'D_hybrid_rrf' in df.experiment.values else df.experiment.iloc[0]
pivot = answerable.pivot(index='id', columns='experiment', values='context_recall')
comparisons = []
rng = np.random.default_rng(42)
if baseline not in pivot.columns:
    print('기준 실험의 성공 결과가 없어 실험 간 개선량은 미측정입니다.')
for name in (pivot.columns if baseline in pivot.columns else []):
    if name == baseline: continue
    paired = pivot[[baseline, name]].dropna()
    delta = (paired[name]-paired[baseline]).to_numpy()
    if not len(delta): continue
    boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
    comparisons.append({'vs':name, 'n':len(delta), 'mean_delta':delta.mean(),
                        'ci_low':np.quantile(boot,.025), 'ci_high':np.quantile(boot,.975),
                        'improved':int((delta>0).sum()), 'worse':int((delta<0).sum())})
display(pd.DataFrame(comparisons))
fig, axes = plt.subplots(1,2,figsize=(12,4))
summary[['context_tokens']].rename(index=lambda x:x.split('_')[0]).plot.bar(ax=axes[0],legend=False,rot=0,color='#2878b5')
axes[0].set_title('Mean context tokens (budget: 6000)')
summary[['retrieval_ms_p50','retrieval_ms_p95']].rename(index=lambda x:x.split('_')[0]).plot.bar(ax=axes[1],rot=0)
axes[1].set_title('Observed retrieval latency (cache may be warm)')
axes[1].set_ylabel('milliseconds')
finish('05_tokens_latency')
''')
md('''## 7. 답변·인용·거절 평가

- `citation_validity`: 반환한 출처 ID와 원문이 실제 컨텍스트에 존재하는 비율. **주장 타당성과는 다릅니다.**
- `citation_support`: 답변의 각 법률 주장을 붙은 인용이 뒷받침하는지 LLM 판정.
- `faithfulness`: 각 법률 주장이 컨텍스트에 의해 뒷받침되는 비율.
- 주장 없는 순수 거절은 충실성/환각 지표를 미측정 처리합니다.
- 외부 호출 실패와 judge 실패는 별도로 세며 거절 성공으로 세지 않습니다.
''')
code('''if not answer_rows:
    display(Markdown('**답변 평가는 미실행입니다.** 전체 API 평가를 실행하면 이 영역이 채워집니다.'))
else:
    ans_summary = answer_summary(answer_rows).set_index('experiment')
    display(ans_summary.round(4))
    metrics = ['correctness','relevance','point_coverage','faithfulness','citation_support']
    fig, ax = plt.subplots(figsize=(11,4))
    ans_summary[metrics].plot.bar(ax=ax, rot=0, ylim=(0,1.1))
    ax.set_title('Automated answer metrics (LLM judge; review required)')
    ax.legend(fontsize=8, loc='lower right')
    finish('06_answer_quality')
    if 'F_rerank_expand' in ans_summary.index:
        r = ans_summary.loc['F_rerank_expand']
        matrix = np.array([[r.true_refusal, r.missed_refusal], [r.false_refusal, r.true_answer]], dtype=int)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(matrix,cmap='Blues')
        ax.set_xticks([0,1], ['Refuse','Answer']); ax.set_yticks([0,1], ['Unanswerable','Answerable'])
        ax.set(xlabel='Predicted',ylabel='Golden',title='Answerability confusion matrix')
        for (i,j),value in np.ndenumerate(matrix): ax.text(j,i,str(value),ha='center',va='center')
        finish('07_refusal_matrix')
    review = [{'id':r['id'],'experiment':r['experiment'],'status':r['status'],
               'expected':r['expected_answerable'],'predicted':r.get('response',{}).get('is_answerable'),
               'answer':r.get('response',{}).get('answer'),
               'judge':r.get('judge',{}).get('explanation'), 'judge_status':r.get('judge_status')}
              for r in answer_rows]
    display(pd.DataFrame(review))
''')
md('''## 8. 정답 조문 직접 제공(Oracle) 진단

Oracle은 `answer_articles`의 전체 조문을 직접 넣습니다. **정답을 사용한 생성 진단이므로 검색 실험 점수에 포함하지 않습니다.**
범위 밖 질문은 Oracle 대상에서 제외하며, 컨텍스트 예산으로 정답 조문이 빠진 경우도 별도 표시합니다.
F와 Oracle의 정답성을 비교하면 검색 근거 부족인지 생성·해석 문제인지 검토할 수 있습니다.
''')
code('''if answer_rows:
    scores = pd.DataFrame([{'id':r['id'],'experiment':r['experiment'],
                           'correctness':r.get('judge',{}).get('correctness'),
                           'oracle_complete':r.get('oracle_complete',True)} for r in answer_rows])
    paired = scores[scores.oracle_complete].pivot(index='id',columns='experiment',values='correctness')
    if {'F_rerank_expand','oracle_gold_articles'}.issubset(paired.columns):
        paired = paired.dropna()
        paired['oracle_minus_retrieved'] = paired.oracle_gold_articles-paired.F_rerank_expand
        display(paired.sort_values('oracle_minus_retrieved',ascending=False))
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(paired.F_rerank_expand,paired.oracle_gold_articles,alpha=.6)
        ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(xlabel='Retrieved-context correctness',ylabel='Oracle-context correctness',
               title='Same questions; different evidence')
        finish('08_oracle')
    else:
        print('Oracle 결과 없음')
''')
md('''## 9. 결과 내보내기와 다음 실험

JSON에는 원문 근거와 질문별 trace, CSV에는 집계 및 문항별 지표, `figures/`에는 PNG가 저장됩니다.
미측정 영역을 0으로 채우지 않고 유지합니다. 세부 근거 평가는 원본을 보존한 별도 검수본에
`required_evidence`를 추가해야 합니다. 모델·청크 크기·K를 바꿀 때는 다른 결과 디렉터리를 사용하세요.

골든셋의 정답 요점도 사람이 검토해야 합니다. 예를 들어 일상 질문의 적용 주체/조건이나
복합 질문의 '전부' 범위는 조문 목록만으로 법률적 완전성을 보장하지 않습니다.
''')
code('''summary.to_csv(RESULT_DIR / 'retrieval_summary.csv')
if answer_rows:
    answer_summary(answer_rows).to_csv(RESULT_DIR / 'answer_summary.csv',index=False)
print('결과 폴더:', RESULT_DIR)
print('차트:', sorted(p.name for p in FIGURES.glob('*.png')))
print('실험:', list(summary.index))
print('검색 오류:', int((df.status != 'ok').sum()))
print('세부 근거 라벨 문항:', audit['evidence_labeled_questions'])
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python (rag-minipjt)',
    'language':'python','name':'python3'}, 'language_info':{'name':'python','version':'3.14'}})
# API-wide outages still produce a useful error report, not a plotting exception.
for index in [8, 9, 11, 15]:
    source = nb.cells[index].source
    nb.cells[index].source = ("if (df.status.eq('ok') & df.answerable).any():\n"
        + '\n'.join('    ' + line for line in source.splitlines())
        + "\nelse:\n    display(Markdown('성공한 검색 결과가 없어 이 차트는 미측정입니다. 오류 표를 확인하세요.'))")
nb.cells[17].source = nb.cells[17].source.replace(
    "    fig, ax = plt.subplots(figsize=(11,4))\n    ans_summary[metrics].plot.bar(ax=ax, rot=0, ylim=(0,1.1))\n    ax.set_title('Automated answer metrics (LLM judge; review required)')\n    ax.legend(fontsize=8, loc='lower right')\n    finish('06_answer_quality')",
    "    if ans_summary.judged.sum() > 0:\n        fig, ax = plt.subplots(figsize=(11,4))\n        ans_summary[metrics].plot.bar(ax=ax, rot=0, ylim=(0,1.1))\n        ax.set_title('Automated answer metrics (LLM judge; review required)')\n        ax.legend(fontsize=8, loc='lower right')\n        finish('06_answer_quality')\n    else:\n        print('Judge 성공 결과 없음: 차트 미측정')")
nb.cells[5].source += "\nif not df.status.eq('ok').any():\n    display(df[['experiment','id','question','error_type']])"
nbf.write(nb, root / 'notebook/rag_evaluation.ipynb')
print('Created notebook/rag_evaluation.ipynb:', len(cells), 'cells')
