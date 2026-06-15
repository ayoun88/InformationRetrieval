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

# ── Elasticsearch 연결 및 검색 함수 ───────────────────────────────────────────
es_username = "elastic"
es_password = "es_password" 

es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=(es_username, es_password),
    ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt"
)

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

def hybrid_retrieve(query_str, size=5, rrf_k=60):
    """
    Sparse(키워드)와 Dense(의미) 검색 결과를 RRF 방식으로 융합합니다.
    Top-K를 5개로 늘려 LLM에게 더 풍부한 정보를 제공합니다.
    """
    sparse_res = sparse_retrieve(query_str, size=50)
    dense_res = dense_retrieve(query_str, size=50)
    
    sparse_hits = sparse_res['hits']['hits']
    dense_hits = dense_res['hits']['hits']
    
    scores = {}
    docs = {}
    
    # 1. Sparse 결과 RRF
    for rank, hit in enumerate(sparse_hits):
        doc_id = hit['_id']
        docs[doc_id] = hit['_source']
        docs[doc_id]['_id_fallback'] = doc_id 
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        
    # 2. Dense 결과 RRF
    for rank, hit in enumerate(dense_hits):
        doc_id = hit['_id']
        docs[doc_id] = hit['_source']
        docs[doc_id]['_id_fallback'] = doc_id
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        
    # 3. 점수 정렬 후 Top-K 반환
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    final_hits = []
    for doc_id, score in sorted_docs[:size]:
        final_hits.append({
            "_source": docs[doc_id],
            "_score": score
        })
        
    return {"hits": {"hits": final_hits}}

# ── LLM 클라이언트 ────────────────────────────────────────────────────────────
os.environ["UPSTAGE_API_KEY"] = "UPSTAGE_API_KEY" 

client = OpenAI(
    api_key=os.environ.get("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1" # 404 에러 원인이었던 부분 수정!
)
llm_model = "solar-pro"

# ── 프롬프트 및 도구 ──────────────────────────────────────────────────────────
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 최대한 활용하여 정확하고 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 "정보가 부족해서 답을 할 수 없습니다."라고 대답한다.
- 한국어로 답변을 생성한다.
"""

# 팀원분이 공유해주신 강력한 프롬프트 적용!
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자의 마지막 메시지가 아래 과학 분야에 해당하는 질문이면 반드시 search api를 호출한다.
  - 물리, 화학, 생물, 천문학, 지구과학, 의학, 영양학, 컴퓨터과학, 전기/전자
  - 자연 현상의 원리, 이유, 특성, 구조, 반응, 차이 등을 묻는 질문
- 자신이 이미 알고 있는 내용이라도 과학 질문이면 반드시 search api를 호출한다.
- 인사, 잡담, 감정 표현, 코딩 질문 등 과학과 무관한 경우에만 검색 없이 답변한다.
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
                    "description": "Final query suitable for use in search from the user messages history."
                }
            },
            "required": ["standalone_query"],
            "type": "object"
        }
    }
}]

# ── RAG 메인 함수 ─────────────────────────────────────────────────────────────
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    
    # [안전장치 1] 에러 발생 시 사용할 사용자의 마지막 질문
    fallback_query = messages[-1]["content"] if messages else ""

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
        print(f"  [Error] Function Calling 실패 (타임아웃 등): {e}")
        # 빈 값 반환 방지: 예외 발생 시 result를 None으로 두고, 아래에서 강제 검색(Fallback)으로 넘김

    # ── Step 1 & 2: 의도 분류 확인 및 검색어 추출 ──────────────────────────────
    standalone_query = fallback_query
    
    if result and result.choices:
        message = result.choices[0].message
        if message.tool_calls:
            # 과학 질문으로 잘 분류됨
            try:
                args = json.loads(message.tool_calls[0].function.arguments)
                standalone_query = args.get("standalone_query", fallback_query)
            except json.JSONDecodeError:
                print("  [Warning] JSON 파싱 에러. Fallback 쿼리 사용.")
            print(f"  [Function Calling] 과학 질문 판단 → 검색어: {standalone_query}")
        else:
            # 과학 무관 일상 대화로 분류됨 -> 바로 답변 반환 (검색 생략)
            print("  [Function Calling] 일상 대화 판단 → 검색 생략")
            response["answer"] = message.content
            return response
    else:
        # API 호출 실패로 result가 None인 경우 -> 0점 방지를 위해 강제로 검색 진행
        print(f"  [Fallback 가동] 분류 실패로 강제 검색 진행 → 검색어: {standalone_query}")

    response["standalone_query"] = standalone_query

    # ── 하이브리드 검색 수행 (Top-K = 5) ──────────────────────────────────────
    search_result = hybrid_retrieve(standalone_query, size=5)

    retrieved_context = []
    for rst in search_result['hits']['hits']:
        content = rst["_source"].get("content", "")
        # docid가 문서에 없으면 _id를 사용하도록 fallback
        docid = rst["_source"].get("docid", rst["_source"].get("_id_fallback", ""))
        
        retrieved_context.append(content)
        response["topk"].append(docid)
        response["references"].append({
            "score": rst["_score"],
            "content": content
        })

    # ── Step 3: 검색 결과 기반 답변 생성 ─────────────────────────────────
    context_str = json.dumps(retrieved_context, ensure_ascii=False)
    # Reference를 포함하여 최종 QA 프롬프트 구성
    qa_messages = messages + [{"role": "user", "content": f"Reference: {context_str}\n\n위 Reference를 바탕으로 질문에 답해줘."}]
    msg_for_qa = [{"role": "system", "content": persona_qa}] + qa_messages
    
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
        # [안전장치 2] 답변 생성마저 실패하면 빈 칸 대신 기본 응답을 넣어 에러 여부를 추적
        response["answer"] = "정보를 처리하는 중 오류가 발생했습니다."

    return response

# ── 평가 실행 ─────────────────────────────────────────────────────────────────
def eval_rag(eval_filename, output_filename):
    print("🚀 평가 파이프라인 시작...")
    with open(eval_filename) as f, open(output_filename, "w", encoding='utf-8') as of:
        for idx, line in enumerate(f):
            j = json.loads(line)
            eval_id = j.get("eval_id", idx)
            print(f'\n[Test {eval_id}] msg: {j["msg"]}')
            
            response = answer_question(j["msg"])
            print(f'  answer: {response["answer"][:50]}...') 
            
            output = {
                "eval_id": eval_id,
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            time.sleep(1) # API 부하 방지용 짧은 대기 시간

if __name__ == "__main__":
    # 문서 색인(create_es_index 등)은 이미 완료되었다고 가정하고 평가 함수만 실행
    eval_rag("../data/eval.jsonl", "v1_final_hybrid_result.csv")
    print("\n✅ 모든 평가가 완료되었습니다!")