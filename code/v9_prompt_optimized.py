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
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")
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


# ── 문서 청킹 (개선: overlap 증가) ───────────────────────────────────────────
def chunk_text_by_sentences(text, max_tokens=400, overlap_sentences=2):
    """
    개선된 청킹 전략:
    - max_tokens: 450 → 400 (안전 마진)
    - overlap: 1 → 2 (문맥 보존 향상)
    """
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
    print("\n=== 1단계: Elasticsearch 인덱싱 시작 (개선된 청킹) ===")
    
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
    print("  [Chunking] 개선된 청킹 작업 시작 (max_tokens=400, overlap=2)...")
    
    for doc in raw_docs:
        content = doc.get("content", "")
        token_len = len(tokenizer.encode(content))
        
        if token_len > 512:
            chunks = chunk_text_by_sentences(content, max_tokens=400, overlap_sentences=2)
            for i, chunk_text in enumerate(chunks):
                new_doc = doc.copy()
                new_doc["content"] = chunk_text
                new_doc["docid"] = f"{doc.get('docid', 'doc')}_chunk{i}" 
                chunked_docs.append(new_doc)
        else:
            chunked_docs.append(doc)
            
    print(f"  [Chunking] 원본 {len(raw_docs)}개 → 청킹 후 {len(chunked_docs)}개")

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


# ── 개선된 Hybrid Retrieve (가중치 적용) ──────────────────────────────────────
def hybrid_retrieve(sparse_query, dense_query, size=20, 
                   sparse_weight=0.65, dense_weight=0.35, rrf_k=55):
    """
    개선 사항:
    - size: 15 → 20 (더 많은 후보)
    - rrf_k: 60 → 55 (조정)
    - Sparse 우선 가중치 (과학 용어 정확 매칭 중요)
    """
    sparse_hits = sparse_retrieve(sparse_query, size=50)['hits']['hits']
    dense_hits  = dense_retrieve(dense_query,   size=50)['hits']['hits']

    scores, docs = {}, {}

    for rank, hit in enumerate(sparse_hits):
        doc_id = hit['_id']
        docs[doc_id] = hit['_source']
        scores[doc_id] = scores.get(doc_id, 0) + sparse_weight * (1.0 / (rrf_k + rank + 1))

    for rank, hit in enumerate(dense_hits):
        doc_id = hit['_id']
        docs[doc_id] = hit['_source']
        scores[doc_id] = scores.get(doc_id, 0) + dense_weight * (1.0 / (rrf_k + rank + 1))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_hits = []
    for doc_id, score in sorted_docs[:size]:
        src = docs[doc_id]
        final_hits.append({"_id": doc_id, "_source": src, "_score": score})

    return {"hits": {"hits": final_hits}}


# ── Cross-encoder Reranking (개선: top_k 증가) ────────────────────────────────
def rerank(query, hits, top_k=7):
    """
    개선: top_k 5 → 7 (더 많은 후보에서 선택)
    """
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


# ── 개선된 프롬프트 ───────────────────────────────────────────────────────────
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
1. 주어진 참고 문서를 **철저히 기반**으로 답변을 생성한다
2. 참고 문서에 없는 내용은 추측하지 않는다
3. 답변은 **간결하고 정확하게** 2-4문장으로 작성한다
4. 전문 용어는 간단한 설명과 함께 제시한다
5. 핵심 답변을 먼저 제시한다 (결론 우선)

## 답변 불가 조건
- 참고 문서에 관련 정보가 전혀 없는 경우
- 문서 내용만으로는 질문에 충분히 답할 수 없는 경우
→ 이 경우 "정보가 부족해서 답을 할 수 없습니다."라고 답변

## 답변 형식
- 불필요한 서론이나 마무리 제거
- 핵심만 간결하게 전달
- 자연스러운 한국어로 작성
- 한국어로 답변을 생성한다
"""

# ★★★ 핵심 개선: Function Calling 프롬프트 ★★★
persona_function_calling = """
## Role: 과학 상식 질문 분류 전문가

## Instructions
다음 조건을 **모두** 만족하는 경우에만 search 함수를 호출한다:

### ✅ Search 함수 호출 (과학 지식 질문)
- 물리, 화학, 생물, 천문학, 지구과학, 의학, 영양학, 컴퓨터과학, 전기/전자 등 **과학 분야의 지식**을 묻는 질문
- 자연 현상, 과학적 원리, 이유, 메커니즘, 특성, 구조, 반응, 차이점 등을 물을 때
- 과학자, 과학적 발견, 과학 용어에 대한 설명을 요청할 때
- "왜", "어떻게", "무엇", "~란", "~에 대해", "~의 원인", "~의 방법" 등으로 **지식을 직접 묻는 질문**

### ❌ Search 함수 호출하지 않음 (일반 대화)
- **인사말**: "안녕", "고마워", "잘 가", "반가워" 등
- **감정 표현**: "기분 좋아", "슬퍼", "무서워", "신나", "~겠다" (단순 감정 토로)
- **잡담**: "날씨 좋다", "배고파", "졸려" 등
- **개인적 의견/평가**: "~좋아", "~싫어", "~재밌어" (지식을 묻지 않음)

### ⚠️ 중요: 함정 케이스 주의!
- **과학 용어를 언급했더라도 지식을 묻지 않으면 검색하지 않는다**
  - ❌ "기억 상실증 걸리면 너무 무섭겠다" → 단순 감정 표현, 검색 안 함
  - ❌ "DNA 연구 너무 재밌어" → 개인 의견, 검색 안 함
  - ✅ "기억 상실증의 원인은?" → 지식 질문, 검색 필요
  - ✅ "DNA가 뭐야?" → 지식 질문, 검색 필요

### 판단 기준
1. "지식을 **직접 요청**하는가?" 확인
2. "~에 대해 알려줘", "~가 뭐야?", "~의 원인은?" 형태면 → 검색
3. 단순히 주제를 언급했을 뿐이면 → 검색하지 않음
4. **확신이 없으면 검색하지 않는다**

### Standalone Query 생성 규칙 (검색하기로 판단한 경우)
- 대화 히스토리가 있으면 이전 맥락을 참고하여 **독립적인 검색 쿼리** 생성
- 대명사(그것, 이거, 저거, 어떤 원인 등)를 구체적인 명사로 교체
- 이전 대화에서 언급된 주제를 명시적으로 포함
- 핵심 키워드 중심으로 간결하게 작성

### 예시
```
입력: "니가 대답을 잘해줘서 너무 신나!"
판단: 감정 표현, 지식 요청 없음 → ❌ 검색하지 않음, 바로 답변

입력: "기억 상실증 걸리면 너무 무섭겠다."
판단: 과학 용어 언급했지만 지식을 묻지 않음, 단순 감정 표현 → ❌ 검색하지 않음

입력: "고마워!"
판단: 감사 인사 → ❌ 검색하지 않음

입력: "기억 상실증의 원인이 뭐야?"
판단: 의학 지식 요청 → ✅ search("기억 상실증의 원인")

입력: [이전] "기억 상실증 걸리면 너무 무섭겠다." 
      [어시스턴트] "네 맞습니다."
      [현재] "어떤 원인 때문에 발생하는지 궁금해."
판단: "원인" 요청 → 지식 질문, 이전 맥락에서 "기억 상실증" 참조 
     → ✅ search("기억 상실증의 발생 원인")

입력: "나무의 분류에 대해 조사해 보기 위한 방법은?"
판단: 생물학 지식 요청 → ✅ search("나무 분류 방법")

입력: "Dmitri Ivanovsky가 누구야?"
판단: 인물 정보 요청 (과학자일 가능성) → ✅ search("Dmitri Ivanovsky")
```
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


# ── RAG 메인 함수 (개선 버전) ─────────────────────────────────────────────────
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    fallback_query = messages[-1]["content"] if messages else ""

    # Step 1: Function Calling (개선된 프롬프트 적용)
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
            print("  [Function Calling] 일반 대화 → 검색 생략")
            response["answer"] = message.content
            return response

    response["standalone_query"] = standalone_query

    # Step 2: HyDE
    hyde_doc = generate_hyde_doc(standalone_query)

    # Step 3: 개선된 Hybrid 검색 (가중치 적용, size 증가)
    combined_query = standalone_query + " " + hyde_doc
    search_result = hybrid_retrieve(
        sparse_query=combined_query, 
        dense_query=combined_query,  
        size=20,             # 15 → 20
        sparse_weight=0.65,  # Sparse 우선
        dense_weight=0.35,
        rrf_k=55             # 60 → 55
    )
    candidates = search_result['hits']['hits']

    # Step 4: 개선된 Reranking (top_k 증가)
    reranked_hits = rerank(standalone_query, candidates, top_k=7)  # 5 → 7

    retrieved_context = []
    seen_docids = set()

    for rst in reranked_hits:
        src = rst["_source"]
        content = src.get("content", "")
        raw_docid = src.get("docid", str(rst.get("_id", "")))
        base_docid = raw_docid.split("_chunk")[0]

        if base_docid not in seen_docids:
            seen_docids.add(base_docid)
            response["topk"].append(base_docid)

        retrieved_context.append(content)
        response["references"].append({"score": rst["_score"], "content": content})

    response["topk"] = response["topk"][:3]

    # Step 5: 개선된 답변 생성
    context_str = json.dumps(retrieved_context, ensure_ascii=False)
    user_query = messages[-1]["content"] if messages else ""
    prompt_with_context = f"다음 검색된 참고 문서를 바탕으로 질문에 정확하게 답해주세요.\n\n[참고 문서]\n{context_str}\n\n[질문]\n{user_query}"
    
    qa_messages = messages[:-1] + [{"role": "user", "content": prompt_with_context}]
    msg_for_qa = [{"role": "system", "content": persona_qa}] + qa_messages

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
    print("\n=== 2단계: RAG 파이프라인 평가 시작 (개선 버전) ===")
    print("개선 사항:")
    print("  - Function Calling 프롬프트 강화 (일반 대화 구분 정확도 향상)")
    print("  - Hybrid 검색: size 15→20, sparse 가중치 0.65, rrf_k 60→55")
    print("  - Reranking: top_k 5→7")
    print("  - 청킹: max_tokens 450→400, overlap 1→2")
    print("  - 답변 생성 프롬프트 개선\n")
    
    with open(eval_filename, "r", encoding="utf-8") as f, \
         open(output_filename, "w", encoding="utf-8") as of:
        
        for idx, line in enumerate(f):
            j = json.loads(line)
            eval_id = j.get("eval_id", idx)
            print(f'\n[Test {eval_id}] Question: {j["msg"]}')

            response = answer_question(j["msg"])
            print(f'  Answer: {response["answer"][:60]}...')
            print(f'  TopK: {response["topk"]}')

            output = {
                "eval_id":          eval_id,
                "standalone_query": response["standalone_query"],
                "topk":             response["topk"],
                "answer":           response["answer"],
                "references":       response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            time.sleep(1)

    print("\n🎉 모든 평가 완료! (개선 버전)")


if __name__ == "__main__":
    # 1. 인덱스 셋업 (개선된 청킹 적용)
    # ⚠️ 청킹 파라미터가 변경되었으므로 재인덱싱 필요!
    # setup_index(index_name="test", data_path="../data/documents.jsonl")
    
    # 2. RAG 평가 실행
    eval_rag("../data/eval.jsonl", "v9_optimized_result.csv")