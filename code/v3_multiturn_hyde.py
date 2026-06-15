import os
import json
import time
import traceback
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
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
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def get_embedding(sentences):
    return model.encode(sentences)

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
# 비밀번호는 환경변수로 관리 권장:
#   export ES_PASSWORD="lLrsQg9KdlTPOa9eybfR"
es_username = "elastic"
es_password = os.environ.get("ES_PASSWORD", "ES_PASSWORD")

es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=(es_username, es_password),
    ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt"
)
print(es.info())


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

def hybrid_retrieve(sparse_query, dense_query, size=5, rrf_k=60):
    """
    Sparse 검색(키워드)과 Dense 검색(의미)을 RRF로 융합한다.
    - sparse_query: BM25 검색에 사용할 쿼리 (standalone_query)
    - dense_query : 임베딩 검색에 사용할 쿼리 (HyDE 가상문서 또는 standalone_query)
    """
    sparse_hits = sparse_retrieve(sparse_query, size=50)['hits']['hits']
    dense_hits  = dense_retrieve(dense_query,  size=50)['hits']['hits']

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


# ── LLM 클라이언트 ────────────────────────────────────────────────────────────
# API 키는 환경변수로 관리 권장:
#   export UPSTAGE_API_KEY="your_key_here"
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

# [개선] 멀티턴 대화 맥락을 반영한 standalone query 생성 지시 추가
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
  → standalone_query: "기억 상실증 발생 원인"  (단독으로 의미가 통하도록 맥락 보완)
"""

# [추가] HyDE용 프롬프트: 주어진 쿼리에 대한 가상의 정답 문서 생성
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


# ── [추가] HyDE: 가상 정답 문서 생성 ─────────────────────────────────────────
def generate_hyde_doc(query):
    """
    쿼리에 대한 가상의 정답 문서를 생성한다.
    이 문서를 Dense 검색 쿼리로 사용하면 실제 관련 문서를 더 잘 찾아낼 수 있다.
    실패 시 원본 쿼리를 그대로 반환한다.
    """
    try:
        result = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": persona_hyde},
                {"role": "user",   "content": query}
            ],
            temperature=0.3,  # 약간의 다양성 허용
            seed=1,
            timeout=20
        ))
        hyde_doc = result.choices[0].message.content.strip()
        print(f"  [HyDE] 가상 문서 생성 완료: {hyde_doc[:60]}...")
        return hyde_doc
    except Exception:
        print("  [HyDE] 생성 실패 → 원본 쿼리로 대체")
        traceback.print_exc()
        return query  # 실패 시 원본 쿼리로 fallback


# ── RAG 메인 함수 ─────────────────────────────────────────────────────────────
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    fallback_query = messages[-1]["content"] if messages else ""

    # ── Step 1: Function Calling으로 의도 분류 + Standalone Query 추출 ─────────
    # tool_calls 있음 → 과학 질문 (멀티턴 맥락 반영한 standalone_query 추출)
    # tool_calls 없음 → 일반 대화 (바로 답변, topk=[] 유지 → MAP 1점)
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
            # 과학 질문 분류됨 → standalone_query 추출
            try:
                args = json.loads(message.tool_calls[0].function.arguments)
                standalone_query = args.get("standalone_query", fallback_query)
            except json.JSONDecodeError:
                print("  [Warning] tool_call JSON 파싱 에러 → fallback 쿼리 사용")
            print(f"  [Function Calling] 과학 질문 → standalone_query: {standalone_query}")
        else:
            # 일반 대화 분류됨 → 검색 없이 바로 반환
            print("  [Function Calling] 일반 대화 → 검색 생략, topk=[]")
            response["answer"] = message.content
            return response
    else:
        # API 호출 실패 → 0점 방지를 위해 강제 검색 진행
        print(f"  [Fallback] 분류 실패 → 강제 검색 진행: {standalone_query}")

    response["standalone_query"] = standalone_query

    # ── Step 2: HyDE 가상 문서 생성 ──────────────────────────────────────────
    # standalone_query → LLM으로 가상 정답 문서 생성 → Dense 검색에 활용
    # Sparse 검색은 원본 standalone_query 사용 (키워드 정확도 유지)
    # Dense 검색은 HyDE 문서 사용 (의미 유사도 향상)
    hyde_doc = generate_hyde_doc(standalone_query)

    # ── Step 3: Hybrid 검색 (Sparse + HyDE-Dense, RRF 융합) ──────────────────
    search_result = hybrid_retrieve(
        sparse_query=standalone_query,  # BM25: 원본 쿼리
        dense_query=hyde_doc,           # Dense: HyDE 가상 문서
        size=5
    )

    retrieved_context = []
    for rst in search_result['hits']['hits']:
        src    = rst["_source"]
        content = src.get("content", "")
        # docid 필드 우선, 없으면 ES 내부 _id 사용
        docid  = src.get("docid", rst.get("_id", ""))

        retrieved_context.append(content)
        response["topk"].append(docid)
        response["references"].append({"score": rst["_score"], "content": content})

    # ── Step 4: 검색 결과 기반 답변 생성 ─────────────────────────────────────
    # [수정] reference를 assistant 메시지로 삽입 (자연스러운 RAG 흐름)
    context_str  = json.dumps(retrieved_context, ensure_ascii=False)
    qa_messages  = messages + [{"role": "assistant", "content": context_str}]
    msg_for_qa   = [{"role": "system", "content": persona_qa}] + qa_messages

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
    print("평가 파이프라인 시작...")
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
            time.sleep(1)  # Rate Limit 예방

    print("\n모든 평가 완료!")


if __name__ == "__main__":
    eval_rag("../data/eval.jsonl", "v3_multiturn_hyde_result.csv")