# 2026-09-23 — LangChain Retrievers

RAG 파이프라인의 **검색(Retrieve)** 단계를 실습한 기록입니다.
기본 `as_retriever()`부터, 검색 결과를 **늘리고(Multi-Query, Ensemble)**, **줄이고(Contextual Compression)**, **바꾸고(Parent/Multi-Vector)**, **거르고(Self-Query, Time-Weighted)**, **다시 배치하는(LongContextReorder)** 방법까지 다룹니다.

```
RAG 파이프라인
[1] Load ──► [2] Split ──► [3] Embed ──► [4] Store ──► [5] Retrieve ──► [6] Generate
                                                        ▲─ 이번 정리 범위 ─▲
```

> 실습 환경: `langchain 1.4.2`, `langchain-classic 1.0.8`, `langchain-core 1.6.3`, `langchain-community 0.4.2`, `langchain-chroma 1.1.0`, `langchain-openai 1.6.2`

---

## 0. 한눈에 고르기

```
검색 결과가 무엇이 문제인가?
├─ 기본으로 충분함                         → VectorStoreRetriever (as_retriever)       [1장]
├─ 질문 표현이 애매해서 놓치는 문서가 있음  → MultiQueryRetriever                       [2장]
├─ 고유명사·키워드가 임베딩으로 잘 안 잡힘  → EnsembleRetriever (BM25 + 벡터)           [3장]
├─ 가져온 문서에 쓸모없는 내용이 많음       → ContextualCompressionRetriever            [4장]
├─ 작은 청크는 잘 찾는데 문맥이 부족함      → ParentDocumentRetriever                   [5장]
├─ 요약/가상질문으로 찾고 원문을 돌려주고 싶음 → MultiVectorRetriever                   [6장]
├─ "2023년 + 평점 4.5 이상" 같은 조건 검색   → SelfQueryRetriever                        [7장]
├─ 최근 문서일수록 우선해야 함               → TimeWeightedVectorStoreRetriever          [8장]
└─ 문서가 많아 LLM이 중간 내용을 놓침        → LongContextReorder (검색 후처리)          [9장]
```

legacy 전환 정리는 [10장](#10-legacy로-바뀐-것들--왜-바뀌었고-지금은-어떻게-쓰나), 노트북에서 발견한 실수는 [11장](#11-노트북에서-발견한-실수--주의점).

---

## 1. VectorStoreRetriever — `as_retriever()`

→ [retriever.ipynb](retriever.ipynb)

벡터스토어를 Retriever 인터페이스(`invoke(query) -> list[Document]`)로 감싸는 가장 기본 형태입니다.

### 매개변수

| 매개변수 | 값 | 의미 |
| --- | --- | --- |
| `search_type` | `"similarity"` (기본) | 코사인/L2 유사도 상위 k개 |
| | `"mmr"` | Maximal Marginal Relevance — 관련성 + **다양성**을 함께 고려해 중복 청크를 줄임 |
| | `"similarity_score_threshold"` | 유사도 점수가 임계값 이상인 것만 반환 (개수는 가변, 0개일 수도 있음) |
| `search_kwargs["k"]` | int (기본 4) | 반환할 문서 수 |
| `search_kwargs["fetch_k"]` | int (기본 20) | MMR 전용. 먼저 후보를 몇 개 뽑은 뒤 그중에서 k개를 고를지 |
| `search_kwargs["lambda_mult"]` | 0~1 (기본 0.5) | MMR 전용. **1 = 관련성만**, **0 = 다양성만** |
| `search_kwargs["score_threshold"]` | 0~1 | threshold 전용. 최소 관련도 점수 |

```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6},
)
docs = retriever.invoke("임베딩(Embedding)은 무엇인가요?")
```

### `ConfigurableField` — 실행 시점에 검색 설정 바꾸기

retriever를 다시 만들지 않고 `config`로 `search_type`/`search_kwargs`를 호출마다 바꿀 수 있습니다.

```python
from langchain_core.runnables import ConfigurableField

retriever = db.as_retriever(search_kwargs={"k": 1}).configurable_fields(
    search_type=ConfigurableField(id="search_type", name="Search Type"),
    search_kwargs=ConfigurableField(id="search_kwargs", name="Search Kwargs"),
)
retriever.invoke("질문", config={"configurable": {"search_kwargs": {"k": 3}}})
```

- `id`: `config["configurable"]`에서 쓸 키 이름
- `name`, `description`: 사람이 보기 위한 설명 (동작에는 영향 없음)

### Upstage 임베딩 — 문서용/쿼리용 모델 분리

`solar-embedding-1-large-passage`(문서) / `solar-embedding-1-large-query`(질문)처럼 **비대칭 임베딩** 모델은 두 개를 따로 써야 합니다. 저장은 passage 모델로, 검색은 query 모델로 벡터를 만든 뒤 `similarity_search_by_vector()`를 호출합니다.

```python
db = FAISS.from_documents(split_docs, UpstageEmbeddings(model="solar-embedding-1-large-passage"))
query_vector = UpstageEmbeddings(model="solar-embedding-1-large-query").embed_query("질문")
db.similarity_search_by_vector(query_vector, k=2)
```

### 워크플로우

```
TextLoader.load() → CharacterTextSplitter.split_documents() → FAISS.from_documents(docs, embeddings)
  → db.as_retriever(search_type, search_kwargs) → retriever.invoke(query)
```

---

## 2. MultiQueryRetriever — 질문을 여러 개로 늘려서 검색

→ [multi_query_retriever.ipynb](multi_query_retriever.ipynb)

LLM이 원 질문을 **관점이 다른 여러 질문**으로 바꾸고, 각각 검색한 결과의 **합집합(중복 제거)** 을 반환합니다. 거리 기반 검색이 질문 표현 하나에 과하게 의존하는 문제를 완화합니다.

### 매개변수 — `MultiQueryRetriever.from_llm(...)`

| 매개변수 | 의미 |
| --- | --- |
| `retriever` | 실제 검색을 수행할 기반 retriever |
| `llm` | 질문 변형을 생성할 모델 |
| `prompt` | 질문 생성 프롬프트 (기본: 3개 생성). **변수명은 `{question}`** 이어야 함 |
| `include_original` | `True`면 원 질문도 검색 목록에 포함 |
| `parser_key` | **deprecated** — 더 이상 사용되지 않음 |

내부적으로 `prompt | llm | LineListOutputParser()` 체인을 만들어 **줄바꿈 단위로 질문을 분리**합니다. 그래서 커스텀 프롬프트도 "한 줄에 질문 하나" 형식으로 출력하게 해야 합니다.

생성된 질문을 보려면 로거 레벨을 올립니다.

```python
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

### 커스텀 프롬프트 사용 (권장 방식)

```python
prompt = PromptTemplate.from_template("""...다섯 개의 질문을 줄바꿈으로 구분해 생성...
#ORIGINAL QUESTION:
{question}""")

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm,
    prompt=prompt,          # ← 체인을 llm 자리에 넣지 말고 prompt 인자로 전달
)
```

> 노트북에서는 `llm=custom_multiquery_chain`처럼 **체인을 llm 자리에** 넣었는데, 이 경우 `기본 프롬프트 | (내 프롬프트 | llm | parser) | LineListOutputParser` 순서가 되어 기본 프롬프트 전체가 내 프롬프트의 `{question}`으로 들어갑니다. 우연히 동작하지만 의도한 구조가 아닙니다. → [11장](#11-노트북에서-발견한-실수--주의점)

### 워크플로우

```
질문 ──► LLM이 N개 질문 생성 ──► 각 질문으로 retriever.invoke ──► 결과 합집합(중복 제거) ──► 반환
```

**비용**: 검색 1번마다 LLM 호출 1번 + 검색 N번.

---

## 3. EnsembleRetriever — BM25(키워드) + 벡터(의미) 하이브리드

→ [ensemble_retriever.ipynb](ensemble_retriever.ipynb)

여러 retriever 결과를 **RRF(Reciprocal Rank Fusion)** 로 합칩니다. 각 문서 점수 = Σ `weight / (rank + c)`.

- **BM25**: 단어가 정확히 일치하는 문서에 강함 (고유명사, 코드명, 오타 없는 키워드)
- **벡터(FAISS)**: 표현이 달라도 의미가 비슷한 문서에 강함

### 매개변수

| 대상 | 매개변수 | 의미 |
| --- | --- | --- |
| `BM25Retriever.from_texts(texts)` / `.from_documents(docs)` | `k` (속성) | 반환 개수. `bm25_retriever.k = 1` 처럼 생성 후 설정 |
| | `preprocess_func` | 토큰화 함수 (기본은 공백 split → **한국어는 형태소 분석기 권장**, 예: Kiwi) |
| `EnsembleRetriever` | `retrievers` | 합칠 retriever 리스트 |
| | `weights` | 각 retriever 가중치 (합 1 권장, 기본 균등) |
| | `c` | RRF 상수 (기본 60). 클수록 순위 차이 영향이 완만해짐 |
| | `id_key` | 문서 동일성 판단 키 (기본: `page_content`로 판단) |

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3],
)
```

`ConfigurableField`로 `weights`를 호출마다 바꿔 비교 실험할 수 있습니다.

```python
ensemble_retriever = EnsembleRetriever(retrievers=[bm25, faiss]).configurable_fields(
    weights=ConfigurableField(id="ensemble_weights", name="Ensemble Weights")
)
ensemble_retriever.invoke(q, config={"configurable": {"ensemble_weights": [1, 0]}})  # BM25만
ensemble_retriever.invoke(q, config={"configurable": {"ensemble_weights": [0, 1]}})  # FAISS만
```

> `BM25Retriever`는 `rank_bm25` 패키지가 필요합니다.

---

## 4. ContextualCompressionRetriever — 검색 결과를 압축/필터링

→ [contextual_compression_retriever.ipynb](contextual_compression_retriever.ipynb)

`base_retriever`로 가져온 문서를 `base_compressor`가 **질문 기준으로 잘라내거나 걸러냅니다**. 프롬프트에 들어가는 토큰을 줄이고 잡음을 제거합니다.

```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,   # 무엇으로 압축할지
    base_retriever=retriever,     # 무엇으로 1차 검색할지
)
```

### Compressor 종류

| Compressor | 동작 | LLM 호출 | 주요 매개변수 |
| --- | --- | --- | --- |
| `LLMChainExtractor.from_llm(llm)` | 문서에서 **질문과 관련된 문장만 추출** (내용을 바꿈) | 문서마다 1회 | `llm`, `prompt` |
| `LLMChainFilter.from_llm(llm)` | 문서를 **통째로 남길지/버릴지** 판단 (내용은 그대로) | 문서마다 1회 | `llm`, `prompt` |
| `EmbeddingsFilter` | 질문–문서 임베딩 유사도가 임계값 미만이면 제거 | 없음 (임베딩만) | `embeddings`, `similarity_threshold`, `k` |
| `EmbeddingsRedundantFilter` | 문서끼리 너무 비슷하면 중복 제거 | 없음 | `embeddings`, `similarity_threshold`(기본 0.95) |
| `DocumentCompressorPipeline` | 위 변환기들을 **순서대로** 적용 | 구성에 따라 | `transformers=[...]` |

### 파이프라인 예시 — 싼 것부터 비싼 것 순서로

```python
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        CharacterTextSplitter(chunk_size=300, chunk_overlap=0),  # 1. 더 잘게 자르고
        EmbeddingsRedundantFilter(embeddings=embeddings),        # 2. 중복 제거
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86),  # 3. 관련성 낮은 것 제거
        LLMChainExtractor.from_llm(llm),                         # 4. 남은 것만 LLM으로 추출
    ]
)
```

> **순서가 비용을 결정합니다.** 임베딩 필터로 먼저 개수를 줄인 뒤 LLM 추출기를 마지막에 두면 LLM 호출 수가 줄어듭니다.
> `similarity_threshold`가 너무 높으면(예: 0.86) 결과가 0개가 될 수 있으니 데이터로 조정해야 합니다.

---

## 5. ParentDocumentRetriever — 작게 찾고 크게 돌려주기

→ [parent_document_retriever.ipynb](parent_document_retriever.ipynb)

- **작은 청크(child)**: 임베딩이 의미를 정확히 담아 **검색이 잘 됨**
- **큰 청크(parent)**: 문맥이 충분해 **답변 생성에 좋음**

child로 검색하고, 매칭된 child의 parent를 돌려줍니다.

### 매개변수

| 매개변수 | 의미 |
| --- | --- |
| `vectorstore` | **child 청크**의 임베딩을 저장 |
| `docstore` | **parent 문서** 원문을 id로 저장 (`InMemoryStore` 등 key-value 저장소) |
| `child_splitter` | child 청크 분할기 (필수) |
| `parent_splitter` | parent 청크 분할기. **생략하면 원본 문서 전체가 parent** |
| `id_key` | child metadata에 parent id를 담는 키 (기본 `"doc_id"`) |
| `search_kwargs` | vectorstore 검색 옵션 |

`add_documents(docs, ids=None, add_to_docstore=True)`
- `ids`: parent id를 직접 지정 (None이면 uuid 자동 생성)
- `add_to_docstore`: False면 vectorstore에만 넣고 docstore에는 안 넣음 (이미 docstore에 있을 때)

### 두 가지 모드

| 모드 | 설정 | 반환되는 것 | 문제점 |
| --- | --- | --- | --- |
| 전체 문서 | `parent_splitter` 없음 | 원본 문서 전체 | 문서가 크면 컨텍스트 폭발 (노트북에서 수천 자) |
| 큰 청크 | `parent_splitter=chunk_size 1000` | 1000자짜리 parent 청크 | 균형 잡힌 기본 선택 |

### 워크플로우

```
add_documents(docs)
  ├─ parent_splitter로 parent 생성 → docstore에 {id: parent} 저장
  └─ 각 parent를 child_splitter로 분할 → child.metadata["doc_id"] = id → vectorstore에 저장

invoke(query)
  └─ vectorstore에서 child 검색 → child의 doc_id 수집(중복 제거) → docstore.mget(ids) → parent 반환
```

`vectorstore.similarity_search()`를 직접 부르면 **child**가, `retriever.invoke()`를 부르면 **parent**가 나옵니다.

---

## 6. MultiVectorRetriever — 문서 하나에 여러 벡터

→ [multi_vetor_retriever.ipynb](multi_vetor_retriever.ipynb)

ParentDocumentRetriever의 일반화 버전입니다. **무엇을 임베딩할지를 직접 정합니다** (작은 청크 / 요약 / 가상 질문). 대신 분할·id 연결·저장을 **수동으로** 해야 합니다.

### 매개변수

| 매개변수 | 의미 |
| --- | --- |
| `vectorstore` | 검색용 벡터(청크/요약/가상질문)를 저장 |
| `byte_store` 또는 `docstore` | 원본 문서 저장소. `byte_store`를 주면 내부적으로 docstore로 감쌈 |
| `id_key` | 검색용 문서 metadata에서 원본 id를 찾을 키 (보통 `"doc_id"`) |
| `search_type` | `SearchType.similarity` / `similarity_score_threshold` / `mmr` |
| `search_kwargs` | `{"k": 1}`, `{"score_threshold": 0.3}` 등 |

### 방법 A — 작은 청크

```python
doc_ids = [str(uuid.uuid4()) for _ in docs]
for i, doc in enumerate(docs):
    for c in child_text_splitter.split_documents([doc]):
        c.metadata[id_key] = doc_ids[i]            # 원본과 연결
        child_docs.append(c)

retriever.vectorstore.add_documents(child_docs)       # 검색용
retriever.docstore.mset(list(zip(doc_ids, docs)))     # 원본
```

### 방법 B — 요약으로 검색

```python
summary_chain = {"doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()
summaries = summary_chain.batch(split_docs, {"max_concurrency": 10})   # 병렬 요약

summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, split_docs)))
```

- `batch(inputs, {"max_concurrency": 10})`: 여러 입력을 동시에 최대 10개씩 병렬 처리

### 방법 C — 가상 질문(Hypothetical Questions)

"이 문서로 답할 수 있는 질문 3개"를 LLM이 만들고 그 질문을 임베딩합니다. 사용자의 질문과 **형태가 같은 것끼리** 비교하므로 매칭이 잘 됩니다. (노트북은 질문 생성까지 진행 — 이후 방법 B와 동일하게 `Document(질문, {doc_id})`로 vectorstore에 넣으면 됩니다.)

> 노트북의 `.bind(functions=..., function_call=...)` + `JsonKeyOutputFunctionsParser` 방식은 legacy입니다. → [10장 ⑥](#-openai-functions-bindfunctions--jsonkeyoutputfunctionsparser)

### 워크플로우

```
원본 docs ──► doc_ids 생성(uuid)
    ├─ 검색용 표현 생성 (청크 / 요약 / 가상질문) + metadata[id_key]=doc_id ──► vectorstore
    └─ (doc_id, 원본) ──► docstore.mset
invoke(query) ──► vectorstore 검색 ──► doc_id ──► docstore에서 원본 반환
```

---

## 7. SelfQueryRetriever — 자연어 질문을 메타데이터 필터로 변환

→ [self_query_retriever.ipynb](self_query_retriever.ipynb)

"카테고리가 메이크업이고 평점 4.5 이상" → LLM이 **검색어 + 필터 조건(StructuredQuery)** 으로 분해 → 벡터스토어 전용 필터 문법으로 번역 → 검색.

### 매개변수

`AttributeInfo(name, description, type)` — LLM에게 메타데이터 필드를 설명
- `description`에 **허용값 목록**을 적어두면(예: `One of ['스킨케어', '메이크업', ...]`) 정확도가 올라감

`SelfQueryRetriever.from_llm(...)`

| 매개변수 | 의미 |
| --- | --- |
| `llm` | 질문 → 구조화 쿼리 변환 모델 (`temperature=0` 권장) |
| `vectorstore` | 검색 대상 |
| `document_contents` | 문서 본문이 무엇인지에 대한 설명 |
| `metadata_field_info` | `AttributeInfo` 리스트 |
| `structured_query_translator` | 벡터스토어별 필터 번역기 (Chroma → `ChromaTranslator()`) |
| `enable_limit` | `True`면 "1개 추천해줘"의 **개수(limit)** 도 질문에서 추출 |
| `search_kwargs` | 기본 검색 옵션 (예: `{"k": 2}`) |

### `enable_limit` 동작 비교

| 설정 | "2023년 상품 추천" | "2023년 상품 1개 추천" |
| --- | --- | --- |
| `enable_limit=False` | 필터 결과 전부 | 필터 결과 전부 (1개 무시) |
| `enable_limit=True, search_kwargs={"k": 2}` | 2개 | 1개 |

### 쿼리 생성기만 따로 보기

```python
from langchain_classic.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt

prompt = get_query_constructor_prompt("Brief summary of a cosmetic product", metadata_field_info)
query_constructor = prompt | llm | StructuredQueryOutputParser.from_components()
query_constructor.invoke({"query": "2023년 스킨케어 제품"})   # → StructuredQuery(query=..., filter=..., limit=...)
```

### 워크플로우

```
질문 ──► query_constructor(프롬프트 | LLM | 파서) ──► StructuredQuery(query, filter, limit)
      ──► ChromaTranslator ──► Chroma where 필터 ──► 벡터 검색 + 필터 적용 ──► 결과
```

---

## 8. TimeWeightedVectorStoreRetriever — 최근성 가중치

→ [timeweighted_vectorstore_retriever.ipynb](timeweighted_vectorstore_retriever.ipynb)

```
score = semantic_similarity + (1.0 - decay_rate) ** hours_passed
```

`hours_passed`는 **마지막으로 접근(검색)된 시각**(`last_accessed_at`) 기준입니다. 생성 시각이 아닙니다. 검색되어 반환된 문서는 `last_accessed_at`이 현재로 갱신되므로 **자주 쓰이는 문서가 계속 살아남습니다**(메모리처럼 동작).

### 매개변수

| 매개변수 | 기본값 | 의미 |
| --- | --- | --- |
| `vectorstore` | — | 임베딩 저장소 (노트북은 빈 FAISS를 직접 생성) |
| `decay_rate` | 0.01 | 0에 가까울수록 오래돼도 점수 유지(≈ 순수 벡터 검색), 1에 가까울수록 **오래된 문서가 빠르게 0점** |
| `k` | 4 | 최종 반환 개수 |
| `search_kwargs` | `{"k": 100}` | 벡터 검색 후보 수 |
| `other_score_keys` | `[]` | metadata의 다른 점수(예: `importance`)를 합산 |
| `default_salience` | None | 벡터 검색에 안 걸린 최근 문서에 줄 기본 유사도 |

| decay_rate | 결과 |
| --- | --- |
| `1e-25` (거의 0) | 하루 지난 문서도 최근성 점수 ≈ 1 → 사실상 **의미 유사도로만** 결정 |
| `0.999` | 하루(24시간) 지난 문서의 최근성 점수 ≈ 0 → **최신 문서 우선** |

### 빈 FAISS 직접 만들기

```python
index = faiss.IndexFlatL2(1536)   # text-embedding-3-small 차원 수
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
#                   임베딩 함수     인덱스  문서저장소          index→docstore id 매핑
```

### `mock_now` — 시간 테스트

```python
from langchain_core.utils.utils import mock_now
with mock_now(datetime.datetime(2026, 9, 23, 0, 0)):
    retriever.invoke("isaac")     # 이 블록 안에서만 datetime.now()가 고정됨
```

> 컨텍스트 매니저라서 `with` 없이 `mock_now(...)`만 호출하면 **아무 효과가 없습니다.** (노트북의 `print(datetime.datetime.now())`가 실제 현재 시각을 찍는 이유)

### `UserWarning: Relevance scores must be between 0 and 1`

`IndexFlatL2`는 **제곱 L2 거리**(정규화 벡터 기준 0~4)를 반환하는데, LangChain 기본 변환식은 `1 - distance / √2`라서 거리가 1.41보다 크면 **음수 점수**가 나옵니다. 검색 순위 자체는 유지되지만 최근성 점수와 더할 때 균형이 틀어집니다. 해결:

```python
vectorstore = FAISS(
    embeddings_model, index, InMemoryDocstore({}), {},
    relevance_score_fn=lambda d: 1.0 - d / 4,    # 제곱 L2(0~4) → 0~1
)
```

---

## 9. LongContextReorder — "Lost in the Middle" 대응

→ [long_context_recorder.ipynb](long_context_recorder.ipynb)

LLM은 긴 컨텍스트의 **처음과 끝은 잘 보고 가운데는 놓치는** 경향이 있습니다. 관련도 높은 문서를 **양 끝**에, 낮은 문서를 **가운데**로 재배치합니다. (매개변수 없음)

```
검색 순서 (관련도순): 1 2 3 4 5 6 7 8 9 10
재배치 후:            2 4 6 8 10 9 7 5 3 1     ← 1위는 맨 끝, 2위는 맨 앞, 꼴찌는 가운데
```

```python
reordered_docs = LongContextReorder().transform_documents(docs)
```

### 체인에 넣기

```python
chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(reorder_documents),
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
)
chain.invoke({"question": "...", "language": "KOREAN"})
```

- `itemgetter("question")`: 입력 dict에서 해당 키만 꺼냄
- `RunnableLambda(fn)`: 일반 함수를 체인 단계로 변환

검색 결과가 적으면(k ≤ 4 정도) 효과가 거의 없고, **k가 클 때** 의미가 있습니다.

---

## 10. legacy로 바뀐 것들 — 왜 바뀌었고 지금은 어떻게 쓰나

### ① `langchain.retrievers` / `langchain.chains` → `langchain_classic`

| 이전 | 현재 |
| --- | --- |
| `from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever, ...` | `from langchain_classic.retrievers import ...` |
| `from langchain.retrievers.multi_query import MultiQueryRetriever` | `from langchain_classic.retrievers.multi_query import ...` |
| `from langchain.chains.query_constructor.base import AttributeInfo` | `from langchain_classic.chains.query_constructor.base import ...` |

**왜**: LangChain 1.0에서 `langchain` 패키지를 **에이전트 중심으로 축소**했습니다. 현재 설치된 `langchain/`에는 `agents`, `chat_models`, `embeddings`, `messages`, `tools` 정도만 남아 있습니다. 기존 체인·retriever·memory 등은 하위 호환을 위해 `langchain-classic`으로 옮겨졌습니다.

**지금은**: 오늘 쓴 고급 retriever들은 여전히 `langchain_classic`에서 가져다 쓰는 것이 정석입니다(동작은 동일). 다만 신규 기능은 이쪽에 추가되지 않으므로, 복잡한 검색 흐름은 **LCEL(`|`)로 직접 조합**하거나 **LangGraph**로 구성하는 방향이 권장됩니다.

### ② `langchain_community` sunset → 전용 통합 패키지

실행 시 뜨는 경고:
```
DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained.
```

**왜**: 수백 개 통합(벡터DB, 로더 등)을 하나의 패키지에서 관리하다 보니 의존성 충돌·유지보수 문제가 컸습니다. 그래서 각 제공자가 **독립 패키지**(`langchain-openai`, `langchain-chroma`, `langchain-upstage` …)로 분리해 관리하는 방식으로 전환했습니다.

| 이전 (community) | 현재 |
| --- | --- |
| `langchain_community.vectorstores.Chroma` (0.2.9부터 deprecated) | `from langchain_chroma import Chroma` ← `long_context_recorder`에서 교체 필요 |
| `langchain_community.embeddings.OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` (이미 사용 중) |
| `FAISS`, `TextLoader`, `WebBaseLoader`, `PyMuPDFLoader`, `LongContextReorder`, `EmbeddingsRedundantFilter`, `ChromaTranslator`, `InMemoryDocstore` | 아직 전용 패키지가 없어 community를 계속 사용 (경고는 뜨지만 동작함). 마이그레이션 안내: langchain-community issue #674 |

### ③ `langchain_classic.*`의 "재수출(re-export) shim" — 원래 위치에서 import

`langchain_classic`의 일부 모듈은 **실제 구현 없이 다른 패키지를 가리키기만** 합니다(`create_importer`로 deprecated 경로 안내). 원래 위치에서 가져오는 편이 명확합니다.

| 노트북에서 쓴 경로 | 실제 위치 (권장) |
| --- | --- |
| `langchain_classic.vectorstores.FAISS` | `langchain_community.vectorstores.FAISS` |
| `langchain_classic.retrievers.BM25Retriever` | `langchain_community.retrievers.BM25Retriever` |
| `langchain_classic.storage.InMemoryStore` | `langchain_core.stores.InMemoryStore` |
| `langchain_classic.docstore.InMemoryDocstore` | `langchain_community.docstore.in_memory.InMemoryDocstore` |
| `langchain_classic.utils.mock_now` | `langchain_core.utils.utils.mock_now` |
| `langchain_classic.output_parsers.openai_functions.JsonKeyOutputFunctionsParser` | `langchain_core.output_parsers.openai_functions` |

### ④ `LLMChain` → LCEL (`prompt | llm | parser`)

**왜**: `LLMChain`은 0.1.17부터 deprecated입니다. 클래스마다 입력/출력 키 규칙이 달라 조합이 어렵고 스트리밍·배치·비동기를 따로 구현해야 했습니다. LCEL은 모든 단계가 `Runnable`이라 `invoke / batch / stream / ainvoke`가 자동으로 생깁니다.

**지금은**: `LLMChainExtractor`, `LLMChainFilter`는 이름에 LLMChain이 남아 있지만, `from_llm()`은 내부에서 **`prompt | llm | parser` LCEL 체인**을 만듭니다. `from_llm()`으로 생성하면 됩니다. 직접 체인을 만들 때는 오늘 요약 체인처럼 LCEL로 작성합니다.

### ⑤ `get_relevant_documents()` → `invoke()`

**왜**: 모든 retriever가 `Runnable` 인터페이스로 통일되면서 `get_relevant_documents` / `aget_relevant_documents`는 deprecated 되었습니다.
**지금은**: `retriever.invoke(query)`, `await retriever.ainvoke(query)`, `retriever.batch([q1, q2])`. (노트북은 이미 `invoke` 사용 ✓)

### ⑥ OpenAI functions (`.bind(functions=...)`) + `JsonKeyOutputFunctionsParser`

**왜**: OpenAI API에서 `functions` / `function_call` 파라미터는 **`tools` / `tool_choice`로 대체된 legacy API**입니다. 또 JSON 스키마를 dict로 손으로 작성해야 해서 실수하기 쉽습니다.

**지금은**: `with_structured_output()`에 Pydantic 모델을 넘기면 tool calling / structured output을 자동으로 쓰고 결과도 객체로 받습니다.

```python
from pydantic import BaseModel, Field

class HypotheticalQuestions(BaseModel):
    questions: list[str] = Field(description="문서로 답할 수 있는 가상 질문 3개")

hypothetical_query_chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("...{doc}")
    | ChatOpenAI(model="gpt-4o-mini").with_structured_output(HypotheticalQuestions)
    | (lambda r: r.questions)
)
```

### ⑦ `MultiQueryRetriever`의 `parser_key`

**왜**: 예전엔 `LLMChain` 출력 dict에서 꺼낼 키를 지정했지만, 이제 LCEL 체인이 `LineListOutputParser`로 바로 `list[str]`을 반환하므로 필요 없어졌습니다.
**지금은**: 넘기지 않습니다. 커스텀 프롬프트는 `prompt=` 인자로 전달합니다.

---

## 11. 노트북에서 발견한 실수 / 주의점

| 노트북 | 내용 | 수정 |
| --- | --- | --- |
| [contextual_compression_retriever.ipynb](contextual_compression_retriever.ipynb) | `LLMChainExtractor` 셀에서 `compression_retriever`가 아니라 **`retriever.invoke`** 를 호출 → 압축 전 결과가 출력됨 | `compression_retriever.invoke(...)` |
| [ensemble_retriever.ipynb](ensemble_retriever.ipynb) | `[Ensemble Retriever]` 아래에서 **`bm25_result`** 를 출력 | `for doc in ensemble_result:` |
| [multi_query_retriever.ipynb](multi_query_retriever.ipynb) | 커스텀 체인을 `llm=` 자리에 전달 (2장 참고) | `from_llm(retriever=..., llm=llm, prompt=prompt)` |
| [timeweighted_vectorstore_retriever.ipynb](timeweighted_vectorstore_retriever.ipynb) | `mock_now(...)`를 `with` 없이 호출 → 효과 없음 | `with mock_now(...):` |
| [timeweighted_vectorstore_retriever.ipynb](timeweighted_vectorstore_retriever.ipynb) | 음수 relevance score 경고 (8장 참고) | `relevance_score_fn` 지정 |
| [long_context_recorder.ipynb](long_context_recorder.ipynb) | `langchain_community.vectorstores.Chroma` (deprecated) | `from langchain_chroma import Chroma` |
| 여러 노트북 | 쿼리 오타 `"Sementic"`, `"isacc"`, `"furit"` — BM25 등 키워드 검색은 오타에 취약해 결과가 달라질 수 있음 | 비교 실험 시 쿼리를 통일 |

---

## 12. Retriever 전체 비교

| Retriever | 추가 LLM 호출 | 인덱싱 비용 | 해결하는 문제 | 단점 |
| --- | --- | --- | --- | --- |
| `as_retriever` (similarity/mmr/threshold) | 없음 | 낮음 | 기본 검색 | 표현 차이·키워드에 약함 |
| MultiQuery | 검색마다 1회 | 낮음 | 질문 표현 편향 | 지연 증가, 결과 수 가변 |
| Ensemble (BM25+벡터) | 없음 | BM25 인덱스 추가 | 키워드 누락 | 가중치 튜닝 필요, BM25는 메모리 상주 |
| Contextual Compression | 문서 수만큼 (LLM 계열) | 낮음 | 잡음·토큰 낭비 | LLM 압축은 느리고 비쌈 |
| ParentDocument | 없음 | 중간 (2단 저장) | 청크 크기 딜레마 | docstore 영속화 필요 |
| MultiVector | 인덱싱 시 (요약/질문 생성) | 높음 | 원문과 검색 표현 분리 | 수동 구성, 인덱싱 비용 큼 |
| SelfQuery | 검색마다 1회 | 메타데이터 설계 | 조건(필터) 검색 | 메타데이터 품질에 의존, 번역기 필요 |
| TimeWeighted | 없음 | 낮음 | 최신성 | 벡터스토어 점수 스케일에 민감 |
| LongContextReorder | 없음 | 없음 | Lost in the middle | k가 작으면 효과 미미 |

실무에서는 하나만 쓰기보다 **조합**합니다. 예:
`Ensemble(BM25 + 벡터)` → `ContextualCompression(EmbeddingsFilter)` → `LongContextReorder` → LLM
