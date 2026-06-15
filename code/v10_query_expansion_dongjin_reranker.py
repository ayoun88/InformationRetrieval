import os
import json
import time
import traceback
import re
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from openai import OpenAI


# ── Rate Limit 재시도 유틸 ────────────────────────────────────────────────────
def call_with_retry(fn, max_retries=5, base_wait=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = (
                "429" in str(e) or
                "too_many_requests" in str(e).lower() or
                "rate limit" in str(e).lower()
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"  [RateLimit] 429 에러 → {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


# ── 모델 초기화 ──────────────────────────────────────────────────────────────
embedding_model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
bi_encoder = SentenceTransformer(embedding_model_name)

# [개선 4] 한국어 특화 Cross-encoder로 교체
cross_encoder = CrossEncoder("Dongjin-kr/ko-reranker")

# 토큰 수 계산을 위한 Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

def get_embedding(sentences):
    return bi_encoder.encode(sentences)

def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'  [Embedding] batch {i} 완료')
    return batch_embeddings


# ── 문서 청킹 (Semantic Chunking) ─────────────────────────────────────────────
def chunk_text_by_sentences(text, max_tokens=450, overlap_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk_sents = []
    current_tokens = 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        sent_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_tokens + sent_tokens > max_tokens and current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))
            current_chunk_sents = current_chunk_sents[-overlap_sentences:]
            current_tokens = sum(
                len(tokenizer.encode(s, add_special_tokens=False))
                for s in current_chunk_sents
            )

        current_chunk_sents.append(sentence)
        current_tokens += sent_tokens

    if current_chunk_sents:
        chunks.append(" ".join(current_chunk_sents))

    return chunks


# ── Elasticsearch 연결 ────────────────────────────────────────────────────────
es_username = "elastic"
es_password = os.environ.get("ES_PASSWORD", "ES_PASSWORD")

es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=(es_username, es_password),
    ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt"
)


# ── Elasticsearch 인덱스 생성 및 데이터 삽입 함수 ───────────────────────
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        print(f"  [ES] 기존 '{index}' 인덱스를 삭제합니다.")
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)
    print(f"  [ES] '{index}' 인덱스가 성공적으로 생성되었습니다.")

def bulk_add(index, docs):
    actions = [{'_index': index, '_source': doc} for doc in docs]
    return helpers.bulk(es, actions)

def setup_index(index_name="test", data_path="../data/documents.jsonl"):
    print("\n=== 1단계: Elasticsearch 인덱싱 시작 (청킹 적용) ===")

    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom", "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed", "filter": ["nori_posfilter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                }
            }
        }
    }
    mappings = {
        "properties": {
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings": {
                "type": "dense_vector", "dims": 768,
                "index": True, "similarity": "l2_norm"
            }
        }
    }
    create_es_index(index_name, settings, mappings)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    chunked_docs = []
    print("  [Chunking] 512 토큰 초과 문서 청킹 작업 시작...")

    for doc in raw_docs:
        content = doc.get("content", "")
        token_len = len(tokenizer.encode(content))

        if token_len > 512:
            chunks = chunk_text_by_sentences(content, max_tokens=450, overlap_sentences=1)
            for i, chunk_text in enumerate(chunks):
                new_doc = doc.copy()
                new_doc["content"] = chunk_text
                new_doc["docid"] = f"{doc.get('docid', 'doc')}_chunk{i}"
                chunked_docs.append(new_doc)
        else:
            chunked_docs.append(doc)

    print(f"  [Chunking] 원본 문서 {len(raw_docs)}개 -> 청킹 후 {len(chunked_docs)}개로 분할 완료.")

    print("  [ES] 문서 임베딩 생성 중...")
    embeddings = get_embeddings_in_batches(chunked_docs)

    index_docs = []
    for doc, embedding in zip(chunked_docs, embeddings):
        doc["embeddings"] = embedding.tolist()
        index_docs.append(doc)

    print("  [ES] 데이터 벌크 삽입 중...")
    ret = bulk_add(index_name, index_docs)
    print(f"  [ES] 인덱싱 완료! (삽입된 문서 수: {ret[0]})")


# ── 검색 함수 ─────────────────────────────────────────────────────────────────
def sparse_retrieve(query_str, size=50):
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size=50):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings", "query_vector": query_embedding.tolist(),
        "k": size, "num_candidates": 100
    }
    return es.search(index="test", knn=knn)

def hybrid_retrieve(sparse_query, dense_query, size=10, rrf_k=60):
    sparse_hits = sparse_retrieve(sparse_query, size=50)['hits']['hits']
    dense_hits  = dense_retrieve(dense_query,   size=50)['hits']['hits']

    scores, docs = {}, {}

    for rank, hit in enumerate(sparse_hits):
        doc_id = hit['_id']
        docs[doc_id] = hit['_source']
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

    for rank, hit in enumerate(dense_hits):
        doc_id = hit['_id']
        docs[doc_id] = hit['_source']
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_hits = []
    for doc_id, score in sorted_docs[:size]:
        src = docs[doc_id]
        final_hits.append({"_id": doc_id, "_source": src, "_score": score})

    return {"hits": {"hits": final_hits}}


# ── Cross-encoder Reranking ───────────────────────────────────────────────────
# [개선 5] Rerank 후보 수를 top_k=7로 줄여 Cross-encoder 정밀도 향상
def rerank(query, hits, top_k=7):
    if not hits:
        return hits
    pairs = [(query, hit["_source"].get("content", "")) for hit in hits]
    scores = cross_encoder.predict(pairs)
    scored_hits = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)

    reranked = []
    for score, hit in scored_hits[:top_k]:
        hit["_score"] = float(score)
        reranked.append(hit)

    print(f"  [Rerank] {len(hits)}개 후보 → top-{top_k} 선별 완료")
    return reranked


# ── LLM 클라이언트 ────────────────────────────────────────────────────────────
os.environ.setdefault("UPSTAGE_API_KEY", "UPSTAGE_API_KEY")
client = OpenAI(
    api_key=os.environ.get("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)
llm_model = "solar-pro3"


# ── 프롬프트 ──────────────────────────────────────────────────────────────────
persona_qa = """
## Role: 과학 상식 전문가
## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 최대한 활용하여 정확하고 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 "정보가 부족해서 답을 할 수 없습니다."라고 대답한다.
- 한국어로 답변을 생성한다.
"""

# [개선 2] 비과학 질문 감지 few-shot 예시 강화
persona_function_calling = """
## Role: 과학 상식 전문가
## Instructions
- 사용자의 마지막 메시지가 아래 과학 분야에 해당하는 질문이면 반드시 search 함수를 호출한다.
  - 물리, 화학, 생물, 천문학, 지구과학, 의학, 영양학, 컴퓨터과학, 전기/전자
  - 자연 현상의 원리, 이유, 특성, 구조, 반응, 차이 등을 묻는 질문
- 아래 '검색 불필요 예시'에 해당하는 경우에는 절대로 search 함수를 호출하지 않고 바로 자연어로 답변한다.
- 인사, 잡담, 감정 표현 등 과학과 무관한 일상 대화에는 search 함수를 호출하지 않고 바로 답변한다.

## 검색 불필요 예시 (절대 search 호출 금지)
- "감사합니다", "고마워요", "잘했어", "대단해"
- "신난다!", "너무 무서워", "재밌겠다", "슬프다"
- "안녕", "잘 지냈어?", "오늘 날씨 어때?"
- "니가 대답을 잘해줘서 너무 신나!"
- 칭찬, 감탄, 인사, 감정 표현이 주를 이루는 메시지

## 검색 필요 예시 (반드시 search 호출)
- "왜 하늘은 파란가요?" → 물리/광학
- "기억 상실증 원인이 뭐야?" → 의학
- "광합성 과정을 설명해줘" → 생물
- "블랙홀은 어떻게 만들어져?" → 천문학
- "DNA 복제 원리가 궁금해" → 생물/분자생물학

## Standalone Query 생성 규칙 (멀티턴 대화 처리)
- 대화 히스토리가 있는 경우, 이전 맥락을 참고하여 마지막 질문만으로도 의미가 완전히 통하는 독립적인 검색 쿼리를 생성한다.
- 예) 이전: "기억 상실증 걸리면 무섭겠다" / 현재: "어떤 원인 때문에 발생하는지 궁금해"
  → standalone_query: "기억 상실증 발생 원인"
"""

persona_hyde = """
## Role: 과학 교과서 작성자
## Instructions
- 주어진 질문에 대해 과학 교과서에 나올 법한 짧은 설명 단락을 한국어로 작성한다.
- 실제 정답이 틀려도 괜찮다. 검색에 활용할 가상의 문서를 생성하는 것이 목적이다.
- 반드시 2~4문장 이내로 간결하게 작성한다.
- 서론 없이 바로 본문 내용만 출력한다.
"""

# [개선 3] 키워드 추출용 프롬프트
persona_keyword = """
## Role: 검색 쿼리 전문가
## Instructions
- 주어진 질문에서 BM25 키워드 검색에 핵심이 되는 한국어 명사/핵심어 3~5개를 추출한다.
- 조사, 어미, 접속사는 제외하고 핵심 단어만 공백으로 구분하여 출력한다.
- 다른 설명 없이 키워드만 출력한다.
- 예) 질문: "광합성은 어떻게 일어나나요?" → 출력: "광합성 엽록체 빛에너지 포도당 이산화탄소"
"""

tools = [{
    "type": "function",
    "function": {
        "name": "search", "description": "search relevant documents",
        "parameters": {
            "properties": {
                "standalone_query": {
                    "type": "string",
                    "description": "Final query suitable for use in search from the user messages history."
                }
            },
            "required": ["standalone_query"], "type": "object"
        }
    }
}]


# ── HyDE: 가상 정답 문서 생성 ─────────────────────────────────────────────────
def generate_hyde_doc(query):
    try:
        result = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": persona_hyde},
                {"role": "user",   "content": query}
            ],
            temperature=0.3, seed=1, timeout=20
        ))
        hyde_doc = result.choices[0].message.content.strip()
        print(f"  [HyDE] 가상 문서 생성 완료")
        return hyde_doc
    except Exception:
        print("  [HyDE] 생성 실패 → 원본 쿼리로 대체")
        return query


# [개선 3] 키워드 추출 함수
def extract_keywords(query):
    try:
        result = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": persona_keyword},
                {"role": "user",   "content": query}
            ],
            temperature=0, seed=1, timeout=15
        ))
        keywords = result.choices[0].message.content.strip()
        print(f"  [Keyword] 추출 완료: {keywords}")
        return keywords
    except Exception:
        print("  [Keyword] 추출 실패 → 원본 쿼리로 대체")
        return query


# ── RAG 메인 함수 ─────────────────────────────────────────────────────────────
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    fallback_query = messages[-1]["content"] if messages else ""

    # Step 1: Function Calling (+ [개선 2] few-shot 강화된 프롬프트 적용)
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    result = None
    try:
        result = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model, messages=msg, tools=tools, tool_choice="auto",
            temperature=0, seed=1, timeout=30
        ))
    except Exception as e:
        print(f"  [Error] Function Calling 실패: {e}")

    standalone_query = fallback_query
    if result and result.choices:
        message = result.choices[0].message
        if message.tool_calls:
            try:
                args = json.loads(message.tool_calls[0].function.arguments)
                standalone_query = args.get("standalone_query", fallback_query)
            except json.JSONDecodeError:
                pass
            print(f"  [Function Calling] 검색 진행 → {standalone_query}")
        else:
            # 비과학 질문: topk를 비워 MAP 1점 확보
            print("  [Function Calling] 일반 대화 → 검색 생략")
            response["answer"] = message.content
            return response

    response["standalone_query"] = standalone_query

    # Step 2: HyDE
    hyde_doc = generate_hyde_doc(standalone_query)

    # Step 3: 키워드 추출 (BM25용) [개선 3]
    keyword_query = extract_keywords(standalone_query)

    # Step 4: Hybrid 검색
    # BM25: 정확한 키워드 매칭을 위해 추출된 핵심 키워드 사용 [개선 3]
    # Dense: 풍부한 의미 매칭을 위해 standalone + HyDE 결합
    combined_query = standalone_query + " " + hyde_doc
    search_result = hybrid_retrieve(
        sparse_query=keyword_query,
        dense_query=combined_query,
        size=30
    )
    candidates = search_result['hits']['hits']

    # Step 5: Reranking — top_k=7로 줄여 정밀도 향상 [개선 5]
    reranked_hits = rerank(standalone_query, candidates, top_k=7)

    retrieved_context = []

    # [개선 1] 청크 → 문서 score max pooling 후 재정렬하여 topk 선정
    doc_scores = {}   # base_docid → best cross-encoder score
    doc_contents = {} # base_docid → 해당 청크의 content (최고 점수 청크)

    for rst in reranked_hits:
        src = rst["_source"]
        content = src.get("content", "")
        raw_docid = src.get("docid", str(rst.get("_id", "")))
        base_docid = raw_docid.split("_chunk")[0]
        score = rst["_score"]

        # max pooling: 같은 문서에서 가장 높은 점수의 청크로 대표
        if base_docid not in doc_scores or score > doc_scores[base_docid]:
            doc_scores[base_docid] = score
            doc_contents[base_docid] = content

        retrieved_context.append(content)
        response["references"].append({"score": score, "content": content})

    # 점수 기준 내림차순으로 base_docid 정렬 → 상위 3개 선택
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    response["topk"] = [docid for docid, _ in sorted_docs[:3]]

    print(f"  [TopK] 선정된 문서: {response['topk']}")

    # Step 6: Answer Generation
    context_str = json.dumps(retrieved_context, ensure_ascii=False)
    user_query = messages[-1]["content"] if messages else ""
    prompt_with_context = (
        f"다음 검색된 참고 문서를 바탕으로 질문에 정확하게 답해주세요.\n\n"
        f"[참고 문서]\n{context_str}\n\n"
        f"[질문]\n{user_query}"
    )

    qa_messages = messages[:-1] + [{"role": "user", "content": prompt_with_context}]
    msg_for_qa  = [{"role": "system", "content": persona_qa}] + qa_messages

    try:
        qaresult = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model, messages=msg_for_qa, temperature=0, seed=1, timeout=30
        ))
        response["answer"] = qaresult.choices[0].message.content
    except Exception as e:
        print(f"  [Error] QA 답변 생성 실패: {e}")
        response["answer"] = "정보를 처리하는 중 오류가 발생했습니다."

    return response


# ── 평가 실행 ─────────────────────────────────────────────────────────────────
def eval_rag(eval_filename, output_filename):
    import csv
    print("\n=== 2단계: RAG 파이프라인 평가 시작 ===")
    with open(eval_filename, "r", encoding="utf-8") as f, \
         open(output_filename, "w", encoding="utf-8", newline="") as of:

        writer = csv.DictWriter(of, fieldnames=["eval_id", "standalone_query", "topk", "answer", "references"])
        writer.writeheader()

        for idx, line in enumerate(f):
            j = json.loads(line)
            eval_id = j.get("eval_id", idx)
            print(f'\n[Test {eval_id}] Question: {j["msg"]}')

            response = answer_question(j["msg"])
            print(f'  Answer: {response["answer"][:60]}...')

            writer.writerow({
                "eval_id":          eval_id,
                "standalone_query": response["standalone_query"],
                "topk":             json.dumps(response["topk"], ensure_ascii=False),
                "answer":           response["answer"],
                "references":       json.dumps(response["references"], ensure_ascii=False)
            })
            time.sleep(1)

    print("\n🎉 모든 평가 완료!")


if __name__ == "__main__":
    # 💡 팁: 이미 이전 단계에서 인덱스를 만들었다면 이 줄은 주석 처리하세요.
    # setup_index(index_name="test", data_path="../data/documents.jsonl")

    # 2. RAG 평가 실행
    eval_rag("../data/eval.jsonl", "v10_query_expansion_dongjin_reranker.csv")