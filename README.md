# 🔬 Information Retrieval - Scientific Knowledge Question Answering
목적 : 질문과 이전 대화 히스토리를 보고 참고할 문서를 검색엔진에서 추출 후, 이를 활용하여 질문에 적합한 대답을 생성하는 RAG 파이프라인 구현

---

## 📂 ReadME Index
[🎯 Project Overview (프로젝트 개요 및 목표)](#project-overview)

[⏱️ Project Duration & 🔧 Tech Stack (기간 및 기술스택)](#projectduration-techstack)

[📊 Data Analysis & Hypothesis (데이터 분석 및 실험 방향성 설정)](#data-analysis)

[🚀 Experimental Progression (실험 과정 및 빌드업)](#experimental-progression)

[🧪 Final SOTA & Experiment Results (핵심 실험 결과 전체)](#final-sota)

[🛠️ Troubleshooting & Engineering (문제 해결 및 인프라 안정화)](#troubleshooting-engineering)

[📈 Retrospective & Future Work (회고 및 향후 계획)](#retrospective-futurework)

---

<a id="project-overview"></a>

## 🎯 Project Overview

### 프로젝트 배경
과학 상식 도메인에서 사용자의 질문과 이전 대화 맥락을 파악해, 가장 관련 있는 문서를 정확하게 탐색하는 **RAG(Retrieval-Augmented Generation) 기반 검색 파이프라인 최적화**

### 핵심 과제
4,272개의 과학 지식 문서 속에서 LLM의 할루시네이션을 방지하고, **가장 관련성 높은 문서 Top-3를 정확하게 추출하는 검색(Retrieval) 파이프라인** 설계 및 고도화

### 핵심 평가 지표

대회 평가 방식은 MAP@3 (Mean Average Precision)을 변형한 방식으로, 두 가지 시나리오를 모두 정확히 처리해야 합니다.

```python
def calc_map(gt, pred):
    for j in pred:
        if gt[j["eval_id"]]:           # 과학 질문 → topk 정답 문서와 MAP@3 계산
            ...
        else:                          # 일반 대화 → topk=[]이면 1점, 뭐라도 있으면 0점
            average_precision = 0 if j["topk"] else 1
```

- **MAP** : 복합적인 과학 지식에 대응하기 위해, 정답 문서를 빠짐없이 Top-3 안에 넣는 능력 극대화
- **MRR** : LLM 토큰 비용을 줄이기 위해, 정답 문서를 최상단(1위)에 노출하는 랭킹 정밀도 최적화
- **의도 분류** : 220개 평가 쿼리 중 20개의 일반 대화 쿼리에 `topk=[]`을 반환해야 MAP 만점 — 의도 분류의 정확도가 전체 점수에 직결됨

---

<a id="projectduration-techstack"></a>

## ⏱️ Project Duration & 🔧 Tech Stack

### ⏱️ Project Duration
- 2026.03.25 ~ 2026.03.31 (5일)

### 🔧 Tech Stack
| Category | Tech Stack |
| :--- | :--- |
| **Language** | Python 3.10 |
| **Search Engine** | Elasticsearch 8.8.0 (Dense & Sparse Hybrid Search) |
| **NLP Tool** | Nori 형태소 분석기 (Elasticsearch 내장), kiwipiepy |
| **Embedding Model** | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (768d), `BAAI/bge-m3` (1024d) |
| **Reranker** | `BAAI/bge-reranker-v2-m3`, `Dongjin-kr/ko-reranker` |
| **LLM** | Upstage Solar (`solar-pro`, `solar-pro3`) |
| **Framework** | HuggingFace Transformers, sentence-transformers |
| **Data Analysis** | Pandas, Matplotlib, WordCloud |
| **Environment** | Ubuntu, GPU Server |
| **Collaboration** | Slack, Zoom, Notion |

---

<a id="data-analysis"></a>

## 📊 Data Analysis & Hypothesis

직관이 아닌 **통계적 근거를 바탕**으로 실험 방향성을 수립하기 위해, 4,272개의 과학 지식 문서와 평가 쿼리의 언어적·구조적 특성을 EDA를 통해 먼저 분석했습니다.



### Insight 1. 쿼리의 짧음 → Dense Retrieval + Hybrid 전략 필요

![Document & Query Length Distribution](https://github.com/user-attachments/assets/eb659f22-40de-47fe-8189-bba0c8ad18b1)

- **분석** : Document 길이는 대부분 **200~400자** 구간에 집중되어 있어 BERT 계열 모델의 Max Sequence Length(512 토큰) 이내에 충분히 들어옵니다. 반면 Query는 **20~30자 내외의 짧은 문장**으로, 정보가 극도로 압축되어 있습니다.
- **실험 방향** : 키워드 매칭(BM25)만으로는 동의어·문맥을 파악하기 어렵기 때문에, 의미적 유사도를 계산하는 **Dense Retrieval의 역할이 핵심**임을 확인하고 하이브리드 검색을 베이스 아키텍처로 설정했습니다. 또한, 1,000자 이상 문서는 일부 존재하므로 해당 문서에 한해서만 청킹을 시도하는 전략을 채택했습니다.



### Insight 2. 중복 데이터가 검색 신뢰도를 위협한다

![Duplicate Documents](https://github.com/user-attachments/assets/163cec70-9a44-4e47-aca5-93bae5e3ab8c)

- **분석** : `docid`는 다르지만 `content`와 `doc_len`이 완전히 일치하는 중복 문서 쌍이 다수 존재함을 확인했습니다. 이는 ko_mmlu의 train/test 분할 과정에서 동일 지식 단위가 중복 생성된 것으로 추정됩니다.
- **실험 방향** : 만약 정답 docid가 `2029`인데 모델이 동일한 내용의 `3194`를 찾아낸다면 오답 처리될 위험이 있습니다. 검색 결과 Top-K 내에서 **중복 내용이 연속으로 등장하지 않도록 제어 로직**이 필요하고, 청킹 시에도 동일 원문 조각들이 Top-K를 독차지하는 밀림 현상을 방지해야 한다는 전략을 수립했습니다.



### Insight 3. 형태소 분석 없이는 BM25가 과학 용어를 놓친다

- **분석** : 워드클라우드 분석 결과 "있습니다, 합니다, 따라서" 같은 의미 없는 기능어가 상위권을 차지했습니다. 이 상태에서 BM25 검색을 수행하면 과학 전문 용어 기반의 정확한 매칭이 불가능합니다. 형태소 분석기로 **명사만 추출**하자 에너지, 지구, 식물, 반응 등 과학 핵심 어휘가 비로소 드러났습니다.
- **실험 방향** : Elasticsearch의 **Nori 형태소 분석기**를 BM25 인덱스에 적용하고, 동시에 "~~하지 않는 이유"처럼 부정형 표현이 단순 키워드로 잘못 토큰화되지 않도록 `stoptags` 설정을 세밀하게 제어하는 전처리 실험을 계획했습니다.

---

### Insight 4. 무분별한 청킹은 검색 품질을 오히려 저하시킨다

- **분석** : 대부분의 문서가 하나의 압축된 지식 단위(200~400자)로 구성되어 있으며, 이 경우 문서를 강제로 쪼개면 파편화된 조각들이 Top-K를 독점하는 **'Top-K 밀림 현상'** 이 발생합니다. 5개로 쪼갠 문서 조각이 Top-3를 차지하면 결과적으로 단 1개의 문서만 찾은 것과 같습니다.
- **실험 방향** : 원문 전체를 색인하는 **No-Chunking 전략**으로 재현율을 극대화하고, 1,000자 이상의 장문 문서에 한해서만 **문장 단위 의미 청킹**을 적용하되, 청킹된 문서의 docid는 `_chunk{i}` 접미사로 관리하고 제출 시 원본 docid로 복원하는 방식으로 설계했습니다.



<a id="experimental-progression"></a>

## 🚀 Experimental Progression

총 12단계의 점진적 실험을 통해 MAP 0.3765 → 0.9167로 성능을 끌어올렸습니다.



### Phase 1. 베이스라인 구축 및 파이프라인 안정화

- **베이스라인 실행** : 기본 Vector DB + LLM 연동 RAG 파이프라인을 구성하고 첫 제출을 완료했습니다(MAP 0.3765). 초기에는 ES 권한 문제, openai-httpx 라이브러리 충돌, API Timeout 등 환경 설정 오류가 집중적으로 발생하여 파이프라인 자체를 안정화하는 것이 첫 번째 과제였습니다.
- **의도 분류 도입 시도** : LLM에게 과학 질문 여부를 판단하게 하고 결과를 JSON으로 반환하도록 지시했으나, 마크다운 코드 블록이나 빈 문자열로 반환되는 JSON 파싱 오류가 빈번하게 발생했습니다(MAP 0.3515로 하락). 이를 해결하기 위해 **Function Calling** 방식으로 전환하여 구조화된 응답을 강제했습니다.

---

### Phase 2. 검색 해상도 극대화 — Hybrid + HyDE + Reranking

- **Hybrid Search + RRF 도입** : BM25(키워드)와 Dense(임베딩) 검색 결과를 Reciprocal Rank Fusion으로 융합했습니다. 동시에 Function Calling 기반 의도 분류와 개선된 프롬프트를 함께 적용하여 MAP이 0.3515 → 0.5614로 크게 향상되었습니다.
- **멀티턴 처리 + HyDE 도입** : 대화 히스토리가 있을 때 맥락을 참고하는 standalone query 생성 프롬프트를 추가하고, LLM으로 가상의 정답 문서를 사전 생성하는 HyDE를 도입했습니다. HyDE 문서는 Dense 검색에만 활용하고, BM25에는 원본 standalone_query를 넘겨 각 검색기의 강점을 분리했습니다. MAP 0.5614 → 0.7894로 점프했습니다.
- **Cross-Encoder Reranking** : Hybrid Search로 후보 10개를 확보한 뒤, `BAAI/bge-reranker-v2-m3`로 (쿼리, 문서) 쌍을 직접 비교하여 최종 Top-3를 선별했습니다(MAP 0.8462). 후보를 넉넉히 확보한 뒤 Reranker가 정밀하게 고르는 2단계 구조가 효과적임을 확인했습니다.



### Phase 3. 형태소 분석기 도입 및 SOTA 탈환 탐색

- **Nori 형태소 분석기 적용 (v5 — 개인 SOTA)** : ES 인덱스와 동일한 Nori 형태소 분석기를 BM25 검색 쿼리에 적용했습니다. 더불어 HyDE 문서도 형태소 분석을 거쳐 키워드를 추출한 뒤 BM25에 함께 전달함으로써, 질문의 핵심 키워드와 HyDE가 생성한 유의어를 동시에 활용했습니다. MAP 0.8462 → **0.8962**로 상승하여 개인 SOTA를 달성했습니다.
- **한계 돌파 탐색** : SOTA 달성 이후 임베딩 모델 교체(bge-m3), RRF 가중치 비대칭 실험, 청킹 전략, 프롬프트 튜닝 등 다양한 변인을 통제하며 추가 실험을 진행했습니다. 대부분 기대와 달리 성능이 하락하여 각 실험을 통해 반직관적 인사이트를 축적했습니다.
- **한국어 Reranker + 키워드 추출 (v10)** : `Dongjin-kr/ko-reranker`로 Reranker를 교체하고, LLM으로 핵심 키워드를 추출하여 BM25에 활용한 결과 MAP 0.9038을 기록했습니다.
- **Multi-Query 확장 + Reranker 앙상블 (v11 — 최종 SOTA)** : 동일 의미의 질문을 LLM으로 다양한 표현으로 N개 생성하여 각각 검색한 뒤 결과를 합산(재현율 향상), `ko-reranker`와 `bge-reranker-v2-m3` 앙상블로 최종 정렬하여 **MAP 0.9167**로 최종 SOTA를 달성했습니다.

---

<a id="final-sota"></a>

## 🧪 Final SOTA & Experiment Results

### 🏆 Final SOTA 아키텍처 (v11)

```
User Query (Multi-turn)
       │
       ▼
┌─────────────────────────┐
│   Function Calling       │  ← 과학 질문 여부 분류 + Standalone Query 추출
│   (Intent Classifier)    │    일반 대화 → topk=[] → MAP 만점 처리
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│         HyDE             │  ← 가상 정답 문서 생성 (Dense Retrieval 품질 향상)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│    Hybrid Search + RRF   │  ← Sparse(Nori BM25) + Dense(bge-m3 임베딩) 1:1 RRF 융합
└──────────┬──────────────┘
           │ Multi-Query 확장으로 후보 다양화
           ▼
┌─────────────────────────┐
│  Reranker Ensemble       │  ← ko-reranker + bge-reranker 앙상블 → Top-3 선별
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Answer Generation      │  ← Solar LLM + Retrieved Context 기반 답변 생성
└─────────────────────────┘
```

---

### 📊 전체 실험 결과 테이블

| 버전 | MAP | MRR | 핵심 변경 | 결과 | 인사이트 |
|------|-----|-----|-----------|------|----------|
| v0_baseline_solar | 0.3765 | 0.3758 | 베이스라인 코드에서 LLM을 Solar로 교체 | — | 환경 설정 오류 다수 발생 (호환성, 결과물 호출 등) |
| v1_classify_topk | 0.3515 | 0.3545 | LLM이 과학 질문 여부 판별 후 JSON 반환 | ⬇️ | 의도 분류는 좋으나, JSON 파싱 오류로 대부분 `topk=[]` 반환 |
| v2_hybridsearch | 0.5614 | 0.5636 | Hybrid(BM25+Dense) 검색 + RRF + 프롬프트 개선 + solar-pro3 | ⬆️ | Function Calling 전환으로 의도 분류 안정화 |
| v3_multiturn_hyde | 0.7894 | 0.7939 | 멀티턴 Standalone Query + HyDE 도입 | ⬆️ | HyDE 가상 문서 → Dense 쿼리, 원본 → Sparse 쿼리 분리가 핵심 |
| v4_rerank | 0.8462 | 0.8455 | Cross-Encoder Reranking (후보 10개 → Top-3) | ⬆️ | `BAAI/bge-reranker-v2-m3`, 후보 확보 후 정밀 선별 효과 입증 |
| **v5_nori_bm25** | **0.8962** | **0.9015** | **Nori 형태소 분석기 BM25 적용 + HyDE 키워드 BM25 투입** | ⬆️ | HyDE 유의어와 원본 키워드를 동시에 BM25에 활용 |
| v6_add_refresh | 0.8159 | 0.8212 | BM25는 standalone_query만 + ES refresh 명시 | ⬇️ | HyDE 키워드가 BM25 검색 범위를 오히려 풍부하게 했음을 역확인 |
| v7_bge_m3 | 0.8886 | 0.8924 | 임베딩 모델 KR-SBERT(768d) → bge-m3(1024d) | ⬇️ | 차원 불일치로 재인덱싱 필수. 모델 교체가 성능 향상을 보장하지 않음 |
| v7-1_cosine | 0.8841 | 0.8894 | bge-m3 유사도 l2_norm → cosine으로 변경 | ⬇️ | 올바른 메트릭 적용에도 bge-m3 효과 없음 → KR-SBERT 복귀 |
| v5-1_weighted | 0.7614 | 0.7667 | Sparse 가중치 1.5배 상향 | ⬇️ | Dense의 의미 매칭 강점이 희석됨. 1:1 균형이 최적임을 재확인 |
| v8_chunking | 0.8902 | 0.8924 | 512 토큰 초과 문서만 문장 단위 청킹 + overlap=1 | ⬇️ | 청킹 조각들이 Top-K 독점 → Top-K 밀림 현상 발생 |
| v8-2_BM25_standalone | 0.8803 | 0.8833 | Reranker 후보 15→30, BM25는 원본 쿼리만 | ⬇️ | Nori+HyDE 키워드 조합이 풍부한 어휘를 제공했음을 재확인 |
| v9_prompt_optimized | 0.8636 | 0.8697 | Few-shot 강화 + RRF 가중치 비대칭 (Sparse 0.65) | ⬇️ | 가중치 비대칭이 성능 저하 주원인. 1:1 유지가 안전 |
| v10_dongjin_reranker | 0.9038 | 0.9061 | 한국어 특화 Reranker + LLM 키워드 추출 BM25 투입 | ⬆️ | Reranker 후보 줄이면 Cross-encoder 분별력 향상 |
| **v11_ensemble** | **0.9167** | **0.9152** | **Multi-Query 확장 + ko-reranker & bge-reranker 앙상블 + 2차 검증** | ⬆️ **최종 SOTA** | 다양한 표현의 쿼리 → 재현율 상승. 앙상블로 오류 감소 |

---

<a id="troubleshooting-engineering"></a>

## 🛠️ Troubleshooting & Engineering

### 1. JSON 파싱 오류 — 의도 분류 안정화

#### 문제 정의
과학 질문 여부를 LLM에게 판단시키고 결과를 JSON으로 받도록 설계했으나, LLM이 마크다운 코드 블록(` ```json ``` `) 또는 빈 문자열 형태로 응답하는 경우가 빈번하게 발생했습니다. 파싱 실패로 인해 대부분의 쿼리가 `topk=[]`로 처리되어 MAP이 0.3765 → 0.3515로 오히려 하락했습니다.

#### 원인 분석
LLM은 출력 형식을 완벽하게 보장하지 않습니다. "JSON만 반환하라"고 지시해도 모델이 설명 문장을 앞에 붙이거나, 코드 블록 마크다운을 감싸는 경우가 발생합니다. 이 경우 `json.loads()`가 즉시 실패하며 예외 처리도 없어 빈 결과가 반환되었습니다.

#### 해결 방안
OpenAI API의 **Function Calling** 방식으로 전환했습니다. `tools` 파라미터에 `search` 함수를 정의하고 `tool_choice="auto"`를 설정하면, LLM이 과학 질문이라 판단할 때만 `tool_calls`를 반환하고 `standalone_query`를 함수 인자로 구조화된 JSON으로 전달합니다. 이로써 파싱 오류 없이 의도 분류와 쿼리 추출을 동시에 안정적으로 처리할 수 있게 되었습니다.

```python
# ❌ Before: JSON 파싱 오류 빈발
result = llm.generate("과학 질문이면 {query: ...} 형식으로 반환하라")
data = json.loads(result)  # 마크다운 블록 감쌈 → 파싱 실패

# ✅ After: Function Calling으로 구조화된 응답 강제
result = client.chat.completions.create(
    model=llm_model, messages=msg,
    tools=tools, tool_choice="auto"
)
if result.choices[0].message.tool_calls:          # 과학 질문
    args = json.loads(tool_call.function.arguments)
    standalone_query = args["standalone_query"]   # 파싱 보장
else:                                             # 일반 대화
    response["answer"] = result.choices[0].message.content
```

#### 인사이트
LLM의 출력에 신뢰성을 부여하려면 프롬프트 지시만으로는 불충분합니다. **Function Calling처럼 API 레벨에서 스키마를 강제**하는 것이 RAG 파이프라인의 안정성을 확보하는 가장 확실한 방법입니다. 특히 파이프라인 중간 단계에서 파싱 실패가 발생하면 하류 전체가 0점을 받는 구조이기 때문에, 구조화된 출력을 보장하는 설계가 필수입니다.



### 2. HyDE와 BM25의 결합 — 반직관적 결과

#### 문제 정의
v5에서 Nori 형태소 분석기로 추출한 HyDE 키워드를 BM25에도 함께 투입하여 SOTA(MAP 0.8962)를 달성했습니다. 그러나 v6에서 "BM25는 짧은 쿼리에 최적화되어 있고 HyDE 장문은 노이즈를 줄 수 있다"는 가설로 BM25에서 HyDE를 제거하고 standalone_query만 넘겼더니 오히려 MAP이 0.8159로 대폭 하락했습니다.

#### 원인 분석
HyDE 문서는 단순한 '긴 텍스트'가 아니라 질문 주제와 관련된 **유의어·동의어를 풍부하게 담고 있는 확장 쿼리**입니다. Nori 형태소 분석기가 이 문서에서 핵심 명사를 추출함으로써 BM25의 키워드 검색 범위를 의도치 않게 확장하는 역할을 하고 있었습니다. 이것이 오히려 검색 재현율을 높이는 효과를 낳았으며, 이를 제거하자 해당 효과가 사라진 것입니다.

#### 해결 방안
HyDE+Nori 조합이 효과적이었던 v5를 다시 SOTA 기준으로 복귀시키고, 두 검색기의 역할을 다음과 같이 명확히 정리했습니다.

```
BM25 쿼리  = Nori 형태소(standalone_query) + Nori 형태소(HyDE 키워드)
Dense 쿼리 = HyDE 문서 전체 (임베딩 검색용)
```

BM25는 형태소 분석기로 걸러진 키워드만 받기 때문에 노이즈가 제어되고, Dense는 HyDE 문서의 의미적 풍부함을 그대로 활용합니다. 두 검색기가 서로 다른 방식으로 HyDE 정보를 활용하는 구조를 확립했습니다.

#### 인사이트
RAG 파이프라인에서 각 모듈의 역할을 이분법적으로 나누는 것보다, **데이터 흐름과 실제 점수 변화를 근거로 최적의 정보 분배 방식을 실험**으로 결정하는 것이 중요합니다. "HyDE는 Dense 전용"이라는 직관적 가설이 틀렸고, 실험 없이 구조를 단정하는 것의 위험성을 배웠습니다.



### 3. 임베딩 모델 업그레이드의 예상외 성능 저하

#### 문제 정의
v5(KR-SBERT, 768d)에서 SOTA를 달성한 후, 더 강력한 다국어 모델인 `BAAI/bge-m3`(1024d)로 교체하면 성능이 오를 것이라 기대했습니다. 그러나 MAP이 0.8962 → 0.8886으로 오히려 하락했고, 코사인 유사도로 메트릭까지 올바르게 수정했음에도 0.8841로 더 하락했습니다.

#### 원인 분석
두 가지 문제가 있었습니다.
1. **차원 불일치** : KR-SBERT(768d)로 구축된 ES 인덱스를 삭제하지 않고 bge-m3로 임베딩 시 `dims=1024` vs `dims=768` 충돌이 발생합니다. 재인덱싱을 하지 않으면 에러 또는 잘못된 검색 결과가 반환됩니다.
2. **도메인 적합성** : bge-m3는 범용 다국어 모델이지만, 이 대회의 코퍼스(4,272개 한국어 과학 문서)에서 KR-SBERT가 더 잘 정렬된 벡터 공간을 형성했을 가능성이 있습니다.

#### 해결 방안
bge-m3 실험을 중단하고 v5(KR-SBERT)로 복귀했습니다. 임베딩 모델 교체 시 **반드시 ES 인덱스를 재생성**해야 한다는 규칙을 확립했고, 이후 실험에서는 변경 사항을 하나씩 격리하여 원인을 추적하는 방식으로 실험을 진행했습니다.

```python
# 모델 교체 시 체크리스트
# 1. ES 인덱스 삭제 및 재생성 (dims 불일치 방지)
# 2. similarity 메트릭 일치 확인 (bge-m3 → cosine, normalize_embeddings=True)
# 3. 모델 단독 변경 후 다른 파라미터 고정 (격리 실험)
```

#### 인사이트
**더 큰 모델 = 더 좋은 성능이 아닙니다.** 특히 특정 언어·도메인에 특화된 소형 모델이 범용 대형 모델보다 성능이 우수한 경우가 많습니다. 모델을 교체할 때는 반드시 인프라 호환성(차원, 메트릭)을 먼저 점검하고, 변수를 하나씩 격리하여 성능 변화의 원인을 명확히 추적해야 합니다.



### 4. 인프라 안정화 — 환경 설정 이슈 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| **ES root 권한 거부** | Elasticsearch는 보안상 root로 실행 불가 | `chown -R 1:1 ./elasticsearch-8.8.0`으로 소유권 이전 후 재실행 |
| **openai + httpx 충돌** | 버전 불일치로 `proxies` 옵션 충돌 | `pip install --upgrade openai httpx` 후 커널 재시작 |
| **API Timeout Error** | Solar 모델 응답 시간 > Python timeout 설정값 | `client.chat.completions.create(timeout=30)` 값 상향 |
| **Rate Limit (429)** | 220개 평가 루프에서 API 호출 과다 | `call_with_retry()` 지수 백오프 + `time.sleep(1)` 추가 |
| **ES 문서 누락** | Bulk insert 후 refresh 전에 검색 시작 | `es.indices.refresh(index="test")` 명시적 호출 추가 |
| **유니코드 출력** | 한글 답변이 raw 유니코드로 출력 | `json.dumps(..., ensure_ascii=False)` 설정 |

---

<a id="retrospective-futurework"></a>

## 📈 Retrospective & Future Work

### 📌 회고
파이프라인을 빠르게 익히고 성능을 끌어올리기 위해 많은 시행착오를 겪었습니다. 강의로 개념을 듣는 것보다 직접 실험하고 점수 변화를 확인하는 과정이 RAG 파이프라인의 작동 원리를 훨씬 빠르게 체득하게 해주었습니다.

팀원들과 EDA 결과, 실험 인사이트, 오류 해결 방법을 슬랙으로 적극적으로 공유하고 좋은 프롬프트를 팀 전체가 활용하는 방식으로 협업한 것이 대회 전반의 성능 향상에 크게 기여했습니다.

### 📌 아쉬운 점
- 대회 초반에 가설을 명확히 세우고 검증하는 절차를 거치지 못하고 시행착오 위주로 실험을 진행한 점이 아쉽습니다.
- 리더보드 일일 제출 횟수 제한으로 인해 아이디어가 있어도 충분히 검증하지 못한 실험들이 있었습니다. 이를 보완하기 위해 로컬 `calc_map` 평가 환경을 구축하는 것을 더 일찍 시작했어야 했습니다.
- 임베딩 모델, 청킹, 가중치 실험 등 다양한 변인을 동시에 바꾸는 경우가 있어 성능 변화의 정확한 원인을 추적하기 어려웠습니다. 앞으로는 **변인 하나씩 격리하는 실험 설계**를 기본 원칙으로 삼겠습니다.

### 📗 향후 계획
- 이번 대회에서 구축한 RAG 파이프라인을 특정 도메인(의료, 법률, 기술 문서 등)에 특화된 챗봇 백엔드에 적용해보고 싶습니다. 도메인 전용 지식 베이스를 잘 구성하면 파인튜닝보다 비용 효율적으로 높은 성능을 낼 수 있다는 가능성을 이번 대회에서 확인했습니다.
- ColBERT, DPR 등 검색 특화 모델을 LLM으로 생성한 pseudo 학습 데이터로 파인튜닝하는 실험을 진행해보고 싶습니다.
- LLM-as-a-Judge 로컬 평가 시스템을 처음부터 구축하여, 제출 횟수 제한 없이 더 많은 실험을 빠르게 순환할 수 있는 환경을 갖추는 것이 목표입니다.
