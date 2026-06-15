import os
import json
import time
import traceback
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
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
# [변경] KR-SBERT(768d, 2021) → bge-m3(1024d, 2024)
# bge-m3: 한국어 포함 100개 언어 지원, MTEB 리더보드 최상위권
# 첫 실행 시 자동 다운로드 (~2.3GB)
bi_encoder   = SentenceTransformer("BAAI/bge-m3")
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")

def get_embedding(sentences):
    # bge-m3는 코사인 유사도 기반 → 정규화 필수
    return bi_encoder.encode(sentences, normalize_embeddings=True)

def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i} 완료')
    return batch_embeddings


# ── Elasticsearch 연결 ────────────────────────────────────────────────────────
es_username = "elastic"
es_password = os.environ.get("ES_PASSWORD", "ES_PASSWORD")

es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=(es_username, es_password),
    ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt"
)
print(es.info())


# ── Elasticsearch 인덱스 생성 및 데이터 삽입 ──────────────────────────────────
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
    print("\n=== 1단계: Elasticsearch 인덱싱 시작 ===")

    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"]
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
                # [변경] bge-m3 출력 차원: 768 → 1024
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
    create_es_index(index_name, settings, mappings)

    index_docs = []
    with open(data_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    print("  [ES] 문서 임베딩 생성 중... (bge-m3, 4200문서 약 5~10분 소요)")
    embeddings = get_embeddings_in_batches(docs)

    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()
        index_docs.append(doc)

    print("  [ES] 데이터 벌크 삽입 중...")
    ret = bulk_add(index_name, index_docs)
    es.indices.refresh(index=index_name)
    print(f"  [ES] 인덱싱 완료! (삽입된 문서 수: {ret[0]})")


# ── 검색 함수 ─────────────────────────────────────────────────────────────────
def sparse_retrieve(query_str, size=50):
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size=50):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)

def hybrid_retrieve(sparse_query, dense_query, size=10, rrf_k=60):
    sparse_hits = sparse_retrieve(sparse_query, size=50)['hits']['hits']
    dense_hits  = dense_retrieve(dense_query,   size=50)['hits']['hits']

    scores = {}
    docs   = {}

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
def rerank(query, hits, top_k=3):
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

persona_function_calling = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 마지막 메시지가 아래 과학 분야에 해당하는 질문이면 반드시 search 함수를 호출한다.
  - 물리, 화학, 생물, 천문학, 지구과학, 의학, 영양학, 컴퓨터과학, 전기/전자
  - 자연 현상의 원리, 이유, 특성, 구조, 반응, 차이 등을 묻는 질문
- 자신이 이미 알고 있는 내용이라도 과학 질문이면 반드시 search 함수를 호출한다.
- 인사, 잡담, 감정 표현 등 과학과 무관한 일상 대화에는 search 함수를 호출하지 않고 바로 답변한다.

## Standalone Query 생성 규칙 (멀티턴 대화 처리)
- 대화 히스토리가 있는 경우, 이전 맥락을 참고하여 마지막 질문만으로도 의미가 완전히 통하는 독립적인 검색 쿼리를 생성한다.
- 예시:
  [user] 기억 상실증 걸리면 너무 무섭겠다.
  [assistant] 네 맞습니다.
  [user] 어떤 원인 때문에 발생하는지 궁금해.
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

tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "search relevant documents",
        "parameters": {
            "properties": {
                "standalone_query": {
                    "type": "string",
                    "description": "Final query suitable for use in search from the user messages history. For multi-turn conversations, rephrase into a self-contained query that includes necessary context."
                }
            },
            "required": ["standalone_query"],
            "type": "object"
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
            temperature=0.3,
            seed=1,
            timeout=20
        ))
        hyde_doc = result.choices[0].message.content.strip()
        print(f"  [HyDE] 가상 문서 생성 완료: {hyde_doc[:60]}...")
        return hyde_doc
    except Exception:
        print("  [HyDE] 생성 실패 → 원본 쿼리로 대체")
        traceback.print_exc()
        return query


# ── RAG 메인 함수 ─────────────────────────────────────────────────────────────
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    fallback_query = messages[-1]["content"] if messages else ""

    # ── Step 1: Function Calling으로 의도 분류 + Standalone Query 추출 ─────────
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    result = None
    try:
        result = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            seed=1,
            timeout=30
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
            print(f"  [Function Calling] 과학 질문 → standalone_query: {standalone_query}")
        else:
            print("  [Function Calling] 일반 대화 → 검색 생략, topk=[]")
            response["answer"] = message.content
            return response
    else:
        print(f"  [Fallback] 분류 실패 → 강제 검색 진행: {standalone_query}")

    response["standalone_query"] = standalone_query

    # ── Step 2: HyDE 가상 문서 생성 ──────────────────────────────────────────
    hyde_doc = generate_hyde_doc(standalone_query)

    # ── Step 3: Hybrid 검색 ───────────────────────────────────────────────────
    combined_query = standalone_query + " " + hyde_doc

    search_result = hybrid_retrieve(
        sparse_query=combined_query,  # BM25: standalone + HyDE 키워드 결합
        dense_query=hyde_doc,         # Dense(bge-m3): HyDE 가상 문서로 의미 검색
        size=10
    )
    candidates = search_result['hits']['hits']

    # ── Step 4: Cross-encoder Reranking → top-3 선별 ─────────────────────────
    reranked_hits = rerank(standalone_query, candidates, top_k=3)

    retrieved_context = []
    for rst in reranked_hits:
        src     = rst["_source"]
        content = src.get("content", "")
        docid   = src.get("docid", rst.get("_id", ""))

        retrieved_context.append(content)
        response["topk"].append(docid)
        response["references"].append({"score": rst["_score"], "content": content})

    # ── Step 5: 검색 결과 기반 답변 생성 ─────────────────────────────────────
    context_str = json.dumps(retrieved_context, ensure_ascii=False)
    qa_messages = messages + [{"role": "assistant", "content": context_str}]
    msg_for_qa  = [{"role": "system", "content": persona_qa}] + qa_messages

    try:
        qaresult = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model,
            messages=msg_for_qa,
            temperature=0,
            seed=1,
            timeout=30
        ))
        response["answer"] = qaresult.choices[0].message.content
    except Exception as e:
        print(f"  [Error] QA 답변 생성 실패: {e}")
        response["answer"] = "정보를 처리하는 중 오류가 발생했습니다."

    return response


# ── 평가 실행 ─────────────────────────────────────────────────────────────────
def eval_rag(eval_filename, output_filename):
    print("\n=== 2단계: RAG 파이프라인 평가 시작 ===")
    with open(eval_filename) as f, open(output_filename, "w", encoding="utf-8") as of:
        for idx, line in enumerate(f):
            j       = json.loads(line)
            eval_id = j.get("eval_id", idx)
            print(f'\n[Test {eval_id}] msg: {j["msg"]}')

            response = answer_question(j["msg"])
            print(f'  answer: {response["answer"][:60]}...')

            output = {
                "eval_id":          eval_id,
                "standalone_query": response["standalone_query"],
                "topk":             response["topk"],
                "answer":           response["answer"],
                "references":       response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            time.sleep(1)

    print("\n모든 평가 완료!")


if __name__ == "__main__":
    # !! bge-m3로 임베딩 모델 교체 → 반드시 재인덱싱 필요 !!
    # dims 768 → 1024로 변경되었기 때문에 기존 인덱스와 호환 안됨
    setup_index(index_name="test", data_path="../data/documents.jsonl")

    eval_rag("../data/eval.jsonl", "v6_bge_m3_result.csv")