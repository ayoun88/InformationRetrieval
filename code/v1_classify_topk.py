
import os
import json
import time
import traceback
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Rate Limit 재시도 유틸 ────────────────────────────────────────────────────
def call_with_retry(fn, max_retries=5, base_wait=5):
    """
    429 RateLimitError 발생 시 지수 백오프로 재시도한다.
    - max_retries: 최대 재시도 횟수
    - base_wait:   첫 대기 시간(초), 이후 2배씩 증가 (5 → 10 → 20 → 40 → 80)
    """
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
                raise  # rate limit 외 에러거나 재시도 초과 시 그냥 올림

# ── 모델 초기화 ──────────────────────────────────────────────────────────────
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# ── 임베딩 ───────────────────────────────────────────────────────────────────
def get_embedding(sentences):
    return model.encode(sentences)

def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings

# ── Elasticsearch ─────────────────────────────────────────────────────────────
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

def delete_es_index(index):
    es.indices.delete(index=index)

def bulk_add(index, docs):
    actions = [{'_index': index, '_source': doc} for doc in docs]
    return helpers.bulk(es, actions)

def sparse_retrieve(query_str, size):
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)

# ── Elasticsearch 연결 ────────────────────────────────────────────────────────
es_username = "elastic"
es_password = "es_password"

es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=(es_username, es_password),
    ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt"
)
print(es.info())

# ── 색인 설정 ─────────────────────────────────────────────────────────────────
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
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

create_es_index("test", settings, mappings)

index_docs = []
with open("../data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)

for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

ret = bulk_add("test", index_docs)
print(ret)

# ── LLM 클라이언트 ────────────────────────────────────────────────────────────
os.environ["UPSTAGE_API_KEY"] = "UPSTAGE_API_KEY"

client = OpenAI(
    api_key=os.environ.get("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

llm_model = "solar-pro3"

# ── 프롬프트 ──────────────────────────────────────────────────────────────────
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# Function calling 프롬프트
# - 과학 질문이면 search 함수를 호출하여 standalone_query를 추출한다.
# - 멀티턴 대화인 경우 이전 메시지 흐름을 참고하여 독립적인 검색 쿼리를 생성한다.
# - 과학 질문이 아닌 일반 대화는 함수를 호출하지 않고 바로 답변한다.
persona_function_calling = """
## Role: 과학 상식 전문가

## Instructions
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search 함수를 호출한다.
- 멀티턴 대화인 경우 대화 맥락을 반영하여 단독으로 의미가 통하는 검색 쿼리를 생성한다.
- 인사, 잡담, 감정 표현 등 과학과 무관한 일상 대화에는 함수를 호출하지 않고 적절한 답변을 생성한다.
"""
tools = [
    {
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
    },
]

# ── RAG 메인 함수 ─────────────────────────────────────────────────────────────
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # ── Step 1: function calling으로 의도 분류 + standalone query 추출 ─────────
    # tool_calls 있음 → 과학 질문 (search 호출 + standalone_query 추출)
    # tool_calls 없음 → 일반 대화 (바로 답변, topk=[] 반환)
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = call_with_retry(lambda: client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice="auto",  # LLM이 과학 질문 여부를 판단하여 search 호출 결정
            temperature=0,
            seed=1,
            timeout=30
        ))
    except Exception:
        traceback.print_exc()
        return response

    # ── Step 2: 과학 질문 → 검색 수행 ────────────────────────────────────────
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        # function.arguments는 API가 JSON 직렬화를 보장 → 파싱 에러 없음
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")
        print(f"  [Function Calling] 과학 질문 → standalone_query: {standalone_query}")

        response["standalone_query"] = standalone_query

        # TODO: 추후 hybrid_retrieve로 교체 예정
        search_result = sparse_retrieve(standalone_query, 3)

        retrieved_context = []
        for i, rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })

        # ── Step 3: 검색 결과 기반 답변 생성 ─────────────────────────────────
        content = json.dumps(retrieved_context, ensure_ascii=False)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = call_with_retry(lambda: client.chat.completions.create(
                model=llm_model,
                messages=msg,
                temperature=0,
                seed=1,
                timeout=30
            ))
            response["answer"] = qaresult.choices[0].message.content
        except Exception:
            traceback.print_exc()

    # ── Step 2 (일반 대화): tool_calls 없음 → 바로 답변, topk=[] 유지 ──────────
    else:
        print("  [Function Calling] 일반 대화 → 검색 생략, topk=[]")
        response["answer"] = result.choices[0].message.content

    return response

# ── 평가 실행 ─────────────────────────────────────────────────────────────────
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        for idx, line in enumerate(f):
            j = json.loads(line)
            print(f'\n[Test {idx}] msg: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'  answer: {response["answer"]}')
            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            time.sleep(1)

eval_rag("../data/eval.jsonl", "v1_classify_topk_result.csv")