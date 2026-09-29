# RAG 직접 구성하기 — Hybrid 검색 + Query Rewriting

[rag_myself_with_ai.ipynb](rag_myself_with_ai.ipynb)의 코드 해설입니다.
PDF 한 권(가상Tech 업무 가이드, 9페이지)으로 **로드 → 분할 → 임베딩 → 저장 → 검색 → 답변 생성**까지 만들고,
검색 품질을 **MMR → Hybrid(BM25+벡터) → Query Rewriting + Hybrid** 순서로 개선해 갑니다.

특히 헷갈리기 쉬운 부분을 자세히 풀었습니다.

- [5. HybridRetriever 클래스](#5-hybridretriever-클래스-bm25--벡터--rrf)
- [6. Query Rewriting](#6-query-rewriting-질문-재작성)
- [7. LCEL 체인 구성](#7-lcel-체인-구성--와-dict가-실제로-하는-일)
- [7-6. 근거를 함께 반환하는 체인](#7-6-근거를-함께-반환하는-체인--traced_rag_chain) — 답이 틀렸을 때 검색 탓인지 LLM 탓인지 가르기
- [7-7. LangSmith 트레이싱](#7-7-langsmith-트레이싱--실행-과정-전체를-웹에서-보기) — 실행 과정 전체를 트리로 보기

```
전체 흐름 (최종 버전)

 사용자 질문 ─┬─► [재작성 LLM] ─► 검색어 ─► [HybridRetriever] ─► 청크 4개 ─► format_docs ─┐
              │                              ├ BM25 (키워드)                                    ├─► prompt ─► LLM ─► 답변 문자열
              │                              └ 벡터 MMR (의미)                                  │
              └──────────────────────────── 원래 질문 그대로 ──────────────────────────────────┘
```

---

## 목차

1. [문서 로드와 분할](#1-문서-로드와-분할)
2. [임베딩](#2-임베딩)
3. [벡터 DB (Chroma)와 중복 방지 id](#3-벡터-db-chroma와-중복-방지-id)
4. [기본 리트리버(MMR)와 첫 번째 RAG 체인](#4-기본-리트리버mmr와-첫-번째-rag-체인)
5. [HybridRetriever 클래스](#5-hybridretriever-클래스-bm25--벡터--rrf)
6. [Query Rewriting](#6-query-rewriting-질문-재작성)
7. [LCEL 체인 구성](#7-lcel-체인-구성--와-dict가-실제로-하는-일)
8. [실험 결과 요약](#8-실험-결과-요약)
9. [알아두면 좋은 점 · 개선 아이디어](#9-알아두면-좋은-점--개선-아이디어)

---

## 1. 문서 로드와 분할

```python
loader = PyPDFLoader("../data/가상Tech_업무가이드.pdf")
docs = loader.load()          # 페이지 1장 = Document 1개 → 9개
```

- `PyPDFLoader`는 **페이지마다 `Document` 하나**를 만듭니다.
- `metadata["page"]`는 **0부터 시작**합니다. 그래서 코드 곳곳에 `page + 1`이 나옵니다(사람이 읽는 페이지 번호로 바꾸려고).

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "다. ", " ", ""],
    add_start_index=True,
)
split_docs = text_splitter.split_documents(docs)   # 9페이지 → 19청크
```

| 설정 | 의미 |
| --- | --- |
| `chunk_size=500` | 청크 하나는 최대 500자 |
| `chunk_overlap=100` | 앞 청크의 끝 100자를 다음 청크 앞에 다시 넣음 → 문장이 경계에서 잘려도 문맥 유지 |
| `separators` | 앞에서부터 차례로 시도: 문단(`\n\n`) → 줄 → 문장(`. `, `다. `) → 단어 → 글자 |
| `add_start_index=True` | 원래 페이지 텍스트에서 청크가 **몇 번째 글자부터 시작하는지**를 `metadata["start_index"]`에 기록 |

> `start_index`는 뒤에서 아주 중요합니다. **`(page, start_index)` 쌍이 청크 하나를 유일하게 가리키는 "주소"** 역할을 해서,
> 벡터 DB id 생성(3장)과 하이브리드 검색의 중복 판별(5장)에 쓰입니다.

---

## 2. 임베딩

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536, chunk_size=100)
```

- 여기서 `chunk_size=100`은 텍스트 분할의 chunk_size와 **다른 뜻**입니다. "API 요청 1번에 텍스트 몇 개를 묶어 보낼지"입니다.
- `embed_documents([...])` → 청크 여러 개를 벡터 리스트로, `embed_query("...")` → 질문 하나를 벡터로 바꿉니다. 둘 다 1536차원.

---

## 3. 벡터 DB (Chroma)와 중복 방지 id

```python
vectorstore = Chroma(
    collection_name="virtualtech_guide",
    embedding_function=embeddings,
    persist_directory="./chroma_db",               # 디스크에 저장 → 다시 실행해도 남아 있음
    collection_metadata={"hnsw:space": "cosine"},  # 코사인 거리로 비교
)

ids = [hashlib.sha256(f'{page}-{start_index}-{page_content}'.encode()).hexdigest() for d in split_docs]

existing_ids = set(vectorstore.get(ids=ids)["ids"])     # DB에 이미 있는 id
new_docs = [d for d, i in zip(split_docs, ids) if i not in existing_ids]
new_ids  = [i for i in ids if i not in existing_ids]
if new_docs:
    vectorstore.add_documents(documents=new_docs, ids=new_ids)
```

**왜 id를 직접 만드나?**
`add_documents`를 id 없이 부르면 실행할 때마다 랜덤 id로 **같은 청크가 또 저장**됩니다(노트북 셀을 3번 실행하면 57개).
청크의 위치+내용으로 SHA-256 해시를 만들면 **같은 청크 → 항상 같은 id**가 되므로,

1. `vectorstore.get(ids=ids)`로 DB에 이미 있는 id를 조회하고
2. 없는 것만 골라서 추가합니다 → **임베딩 API도 새 청크에 대해서만 호출**됩니다(비용 절약 = 임베딩 캐시 역할).

문서 내용이 바뀌면 해시도 바뀌므로 새 청크로 들어갑니다. (단, 예전 청크는 자동으로 지워지지 않는다는 점은 주의)

```python
vectorstore.similarity_search_with_score(query, k=3)          # 점수 = 코사인 "거리" → 낮을수록 비슷
vectorstore.similarity_search(query, k=2, filter={"page": 7})  # 8페이지에서만 검색 (page는 0부터라 7)
```

---

## 4. 기본 리트리버(MMR)와 첫 번째 RAG 체인

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.7},
)
```

MMR(Maximal Marginal Relevance)은 두 단계로 동작합니다.

1. 질문과 가까운 후보를 `fetch_k=12`개 먼저 가져옴
2. 그중에서 "질문과 관련 있으면서 **이미 뽑은 청크와는 다른**" 청크를 하나씩 골라 `k=4`개를 채움
   (`lambda_mult`가 1에 가까우면 관련성 중심, 0에 가까우면 다양성 중심)

`chunk_overlap=100` 때문에 이웃 청크끼리 내용이 겹치는데, 그냥 similarity로 뽑으면 거의 같은 청크가 여러 개 나옵니다. MMR이 그걸 걸러 줍니다.

첫 번째 RAG 체인(`rag_chain`)의 구조는 최종 체인과 같으므로 [7장](#7-lcel-체인-구성--와-dict가-실제로-하는-일)에서 함께 설명합니다.

---

## 5. HybridRetriever 클래스 (BM25 + 벡터 + RRF)

### 5-1. 왜 필요한가

| 검색 방식 | 잘하는 것 | 못하는 것 |
| --- | --- | --- |
| **BM25** (키워드) | `LearnHub`, `15만원`처럼 **글자가 똑같이** 나오는 고유명사·숫자 | 표현이 다르면 전혀 못 찾음 ("아파서" ≠ "병가") |
| **벡터** (의미) | 표현이 달라도 **뜻이 비슷하면** 찾음 | 처음 보는 고유명사·코드명은 뜻을 몰라 약함 |

둘을 같이 돌리고 결과를 합치면 서로의 약점을 메워 줍니다. 이게 하이브리드 검색입니다.
LangChain에는 원래 `EnsembleRetriever`가 있지만 1.x에서는 레거시 패키지(`langchain_classic`)로 빠졌기 때문에, **`BaseRetriever`를 상속해 직접 만들었습니다.**

### 5-2. 설정값

```python
HYBRID_CONFIG = {
    "k": 4,               # 최종 반환 개수
    "candidate_k": 5,     # BM25, 벡터 각각에서 가져올 후보 수
    "bm25_weight": 0.5,
    "vector_weight": 0.5,
    "rrf_k": 60,          # RRF 상수
}
```

전체 흐름: **BM25 후보 최대 5개 + 벡터 후보 5개 → 합쳐서 점수 매기기 → 상위 4개 반환**

### 5-3. 토크나이저 `tokenize()` — BM25가 볼 "단어"를 만드는 함수

BM25는 "질문의 단어가 청크에 몇 번 나오는가"로 점수를 매깁니다. 그래서 **단어를 어떻게 자르느냐**가 성능을 좌우합니다.

한국어를 띄어쓰기로만 자르면 `"휴가는"`, `"휴가를"`, `"휴가"`가 전부 다른 단어가 되어 매칭이 안 됩니다.
그래서 형태소 분석기 **Kiwi**로 조사·어미를 떼어 냅니다.

```python
def tokenize(text: str) -> list[str]:
    return [
        t.form.lower()                          # 형태소 원형을 소문자로 (LearnHub → learnhub)
        for t in kiwi.tokenize(text)
        if t.tag[0] in ("N", "V", "S", "X")     # 품사 태그 첫 글자로 거르기
        and t.tag != "SF"                       # 단, 마침표·물음표(SF)는 제외
    ]
```

품사 태그 첫 글자 의미:

| 첫 글자 | 품사 | 예 | 남김? |
| --- | --- | --- | --- |
| `N` | 명사류 (NNG 일반명사, NNP 고유명사, NNB 의존명사, NR 수사, NP 대명사) | 출장, 숙박비, 만, 원 | O |
| `V` | 용언 (VV 동사, VA 형용사, VX 보조용언, VCP 이다) | 쓰, 아프, 하 | O |
| `S` | 기호·외국어·숫자 (SL 외국어, SN 숫자, SF 마침표 …) | LearnHub, 15 | O (SF만 제외) |
| `X` | 접사·어근 (XSV "-하다"의 하, XR 어근) | 하 | O |
| `J` | 조사 | 는, 을, 가 | **X** |
| `E` | 어미 | 나요, 어요 | **X** |
| `M` | 부사·관형사 | 언제, 매우 | **X** |
| `W` | 웹 요소 (W_HASHTAG, W_URL, W_EMAIL …) | #help-urgent | **X** ← 아래 주의 참고 |

실제로 돌려 본 결과:

```
"출장 숙박비 한도가 15만원이에요?"
  Kiwi 전체 : 출장/NNG 숙박비/NNG 한도/NNG 가/JKS 15/SN 만/NR 원/NNB 이/VCP 에요/EF ?/SF
  남는 토큰 : ['출장', '숙박비', '한도', '15', '만', '원', '이']

"LearnHub에서는 뭘 신청하나요?"
  남는 토큰 : ['learnhub', '뭐', '신청', '하']

"#help-urgent 채널은 언제 쓰나요?"
  Kiwi 전체 : #help-urgent/W_HASHTAG 채널/NNG 은/JX 언제/MAG 쓰/VV 나요/EF ?/SF
  남는 토큰 : ['채널', '쓰']          ← '#help-urgent'가 사라짐!
```

> **주의 (현재 코드의 숨은 약점)**: Kiwi는 `#help-urgent`를 해시태그(`W_HASHTAG`)로 인식하는데, 필터에 `W`가 없어서 **BM25에서 이 키워드가 통째로 빠집니다.**
> 실험에서 #help-urgent 질문이 맞은 건 벡터 검색 덕분입니다. 고치려면 조건에 `"W"`를 추가하면 됩니다:
> `t.tag[0] in ("N", "V", "S", "X", "W")`

같은 `tokenize`를 **문서 쪽(BM25 인덱스 만들 때)과 질문 쪽(검색할 때) 모두**에 써야 토큰이 서로 맞습니다. 코드도 그렇게 되어 있습니다.

### 5-4. 클래스 선언부 — "필드만 적었는데 왜 `__init__`이 없지?"

```python
class HybridRetriever(BaseRetriever):
    documents: list[Document]
    vector_retriever: BaseRetriever
    bm25: BM25Okapi
    config: dict
```

`BaseRetriever`는 내부적으로 **Pydantic `BaseModel`** 입니다. Pydantic 클래스에서는

- 클래스 본문에 `이름: 타입`만 적으면 그게 **필드**가 되고
- `__init__`을 자동으로 만들어 줍니다 → `HybridRetriever(documents=..., bm25=..., ...)`처럼 **키워드 인자로만** 생성
- 생성할 때 타입 검사도 합니다. `BM25Okapi`처럼 Pydantic이 모르는 타입도 `BaseRetriever`가 `arbitrary_types_allowed=True`로 설정해 두어서 받아 줍니다.

그래서 일반 파이썬 클래스처럼 `def __init__(self, documents, ...): self.documents = documents` 를 쓸 필요가 없습니다.

| 필드 | 들어가는 값 | 역할 |
| --- | --- | --- |
| `documents` | `split_docs` (19청크) | BM25 점수 배열의 i번째 = `documents[i]`로 되돌려 찾기 위한 원본 목록 |
| `vector_retriever` | Chroma MMR 리트리버 | 의미 검색 담당 |
| `bm25` | `BM25Okapi(토큰화된 19청크)` | 키워드 검색 담당 (미리 만들어 둔 인덱스) |
| `config` | `HYBRID_CONFIG` | 후보 수, 가중치, rrf_k |

### 5-5. 왜 `_get_relevant_documents` 하나만 구현하면 되나

`BaseRetriever`는 **Runnable**이라서 `invoke`, `batch`, `ainvoke`, `|` 연결, 콜백/트레이싱을 이미 다 갖고 있습니다.
사용자가 `hybrid_retriever.invoke("질문")`을 부르면 내부에서 대략 이렇게 동작합니다.

```
invoke(query)
 ├─ 콜백 매니저 준비, "retriever 시작" 이벤트 기록 (LangSmith 트레이싱 등)
 ├─ self._get_relevant_documents(query, run_manager=run_manager)   ← 우리가 만든 부분
 └─ "retriever 종료" 이벤트 기록, 결과 반환
```

즉 **"질문 문자열을 받아서 Document 리스트를 돌려주는 로직"만 채우면** 나머지는 부모 클래스가 해 줍니다.
시그니처의 `*`는 "그 뒤 인자(`run_manager`)는 키워드로만 받는다"는 뜻이고, 부모 클래스가 그렇게 호출하기 때문에 형태를 맞춘 것입니다.

### 5-6. 검색 로직 한 줄씩

#### 1단계 — BM25 키워드 검색

```python
scores = self.bm25.get_scores(tokenize(query))
```

- 질문을 토큰화해서 넘기면 **19개 청크 각각의 점수 배열**이 돌아옵니다. 예: `[0.0, 3.2, 0.0, 7.9, ...]` (길이 19)
- 배열의 순서 = BM25를 만들 때 넣은 `split_docs` 순서. 그래서 **i번째 점수 = `self.documents[i]`의 점수**입니다.
- 질문 토큰이 하나도 안 나오는 청크는 0점.

```python
bm25_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: cfg["candidate_k"]]
```

"점수 높은 순으로 **인덱스**를 정렬해서 앞의 5개"를 뜻합니다. 풀어 쓰면:

```python
indices = [0, 1, 2, ..., 18]
indices.sort(key=lambda i: scores[i], reverse=True)   # 점수 큰 인덱스가 앞으로
bm25_top = indices[:5]                                # 예: [3, 1, 11, 7, 0]
```

값(점수)이 아니라 **인덱스**를 정렬하는 이유는, 나중에 `self.documents[i]`로 원본 청크를 꺼내야 하기 때문입니다.

```python
bm25_docs = [self.documents[i] for i in bm25_top if scores[i] > 0]
```

상위 5개 중 **점수가 0인 청크는 버립니다.** 키워드가 하나도 안 겹치는데 순위만 5위 안에 든 청크(예: 전체적으로 매칭이 거의 없는 질문)를
"BM25 후보"로 쳐주면 안 되기 때문입니다. 그래서 `bm25_docs`는 5개보다 적을 수도 있습니다.

#### 2단계 — 벡터 검색

```python
vector_docs = self.vector_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
```

- 그냥 `self.vector_retriever.invoke(query)`만 해도 결과는 똑같습니다.
- `run_manager.get_child()`를 콜백으로 넘기면 트레이싱 화면에서
  ```
  HybridRetriever
   └─ VectorStoreRetriever   ← 하위 단계로 표시됨
  ```
  처럼 **부모-자식 관계로 기록**됩니다. 안 넘기면 두 검색이 서로 관계없는 별개 실행으로 찍힙니다.
- 이 벡터 리트리버는 생성할 때 `k=5, fetch_k=15`의 MMR로 설정했습니다(5-8 참고).

#### 3단계 — RRF(Reciprocal Rank Fusion)로 합치기

**문제**: BM25 점수(예: 7.9)와 벡터 거리(예: 0.32)는 단위가 전혀 달라서 그냥 더할 수 없습니다.
**해결**: 점수는 버리고 **순위만** 씁니다. 순위는 둘 다 1, 2, 3 … 으로 같은 단위이기 때문입니다.

```
청크의 RRF 점수 = Σ (각 검색 결과에서)  weight / (rrf_k + 순위)
```

```python
fused, doc_by_key = {}, {}
for docs_, weight in ((bm25_docs, 0.5), (vector_docs, 0.5)):   # BM25 목록, 벡터 목록을 차례로
    for rank, d in enumerate(docs_, start=1):                   # 순위는 1부터
        key = (d.metadata["page"], d.metadata["start_index"])   # 청크 "주소"
        fused[key] = fused.get(key, 0) + weight / (60 + rank)   # 점수 누적
        doc_by_key[key] = d                                     # 주소 → Document 기억
```

- `fused`: `{청크 주소: 누적 RRF 점수}`
- `fused.get(key, 0)`: 처음 보는 청크면 0에서 시작, **이미 BM25에서 나온 청크가 벡터에서 또 나오면 점수가 더해집니다.** ← 핵심
- **왜 `key`를 `(page, start_index)`로 만드나?**
  BM25 쪽 Document는 `split_docs`의 객체이고, 벡터 쪽 Document는 Chroma에서 새로 꺼낸 **다른 객체**입니다.
  내용이 같아도 `d1 is d2`는 False이고 Document는 dict 키로 쓸 수도 없습니다(해시 불가).
  그래서 "몇 페이지의 몇 번째 글자부터 시작하는 청크"라는 **값**으로 같은 청크인지 판별합니다.
- `doc_by_key`: 최종 결과를 돌려줄 때 주소로 Document를 다시 찾기 위한 표. 같은 청크가 두 번 나오면 나중 것(벡터 쪽)으로 덮어쓰지만 내용은 같으니 상관없습니다.

```python
ranked = sorted(fused, key=fused.get, reverse=True)[: cfg["k"]]
return [doc_by_key[key] for key in ranked]
```

- `sorted(fused, ...)`: dict를 정렬하면 **키(청크 주소)들**이 정렬됩니다.
- `key=fused.get`: 각 주소의 점수로 비교 → 점수 높은 순 → 앞의 4개
- 주소 목록을 다시 Document 목록으로 바꿔 반환

### 5-7. RRF 숫자로 직접 계산해 보기

`weight=0.5`, `rrf_k=60`일 때 순위별 점수:

| 순위 | 0.5 / (60 + 순위) |
| --- | --- |
| 1위 | 0.5 / 61 = **0.00820** |
| 2위 | 0.5 / 62 = 0.00806 |
| 3위 | 0.5 / 63 = 0.00794 |
| 4위 | 0.5 / 64 = 0.00781 |
| 5위 | 0.5 / 65 = 0.00769 |

예시 상황 (청크를 A~G로 표기):

```
BM25 결과 : A(1위) B(2위) C(3위)            ← 점수 0인 청크는 버려져서 3개만
벡터 결과 : D(1위) B(2위) E(3위) C(4위) F(5위)
```

| 청크 | BM25 기여 | 벡터 기여 | 합계 | 최종 순위 |
| --- | --- | --- | --- | --- |
| B | 0.00806 (2위) | 0.00806 (2위) | **0.01613** | 1 |
| C | 0.00794 (3위) | 0.00781 (4위) | **0.01575** | 2 |
| A | 0.00820 (1위) | – | 0.00820 | 3 |
| D | – | 0.00820 (1위) | 0.00820 | 4 |
| E | – | 0.00794 (3위) | 0.00794 | 5 (탈락) |
| F | – | 0.00769 (5위) | 0.00769 | 6 (탈락) |

알 수 있는 것:

1. **양쪽 모두에 나온 청크(B, C)는 거의 무조건 1·2위**가 됩니다. 한쪽 1위(0.0082)보다 양쪽 하위권 합(0.0157)이 두 배 가까이 크기 때문입니다.
2. 한쪽에만 나온 청크끼리는 순위 차이가 아주 작습니다(1위 0.0082 vs 5위 0.0077). `rrf_k=60`이 크기 때문에 "순위 차이의 영향이 완만"한 것입니다.
   `rrf_k`를 작게(예: 1) 하면 1위 0.25, 5위 0.083으로 차이가 커져서 각 검색의 1위를 훨씬 강하게 밀어 줍니다.
3. A와 D처럼 동점이면 `sorted`는 원래 순서를 유지(안정 정렬)하는데, `fused`에 BM25가 먼저 들어가므로 **BM25 쪽이 앞**에 옵니다.

**`candidate_k`를 8이 아니라 5로 한 이유**가 여기서 나옵니다.
전체 청크가 19개뿐인데 양쪽에서 8개씩 가져오면 두 목록이 대부분 겹칩니다. 그러면 1번 규칙 때문에 "겹친 청크들"이 상위 4자리를 다 차지하고,
**BM25에서만 1위인 진짜 정답 청크(LearnHub 질문의 경우)가 5위 밖으로 밀려납니다.** 후보를 5개로 줄이면 겹침이 줄어서 이런 청크도 살아남습니다.

### 5-8. 인스턴스 생성

```python
hybrid_retriever = HybridRetriever(
    documents=split_docs,
    vector_retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},   # candidate_k=5, fetch_k=5*3
    ),
    bm25=BM25Okapi([tokenize(d.page_content) for d in split_docs]),   # 19청크를 토큰화해 인덱스 생성
    config=HYBRID_CONFIG,
)
```

- 4장의 `retriever`(k=4)를 재사용하지 않고 **k=5짜리 벡터 리트리버를 새로 만든** 이유: 하이브리드에서는 벡터 쪽도 "후보"를 5개 가져와야 하기 때문입니다.
- `BM25Okapi(...)`는 생성 시점에 각 단어가 몇 개 청크에 나오는지(IDF), 청크 길이 등을 **미리 계산**해 둡니다. 그래서 검색할 때는 `get_scores`만 부르면 됩니다.
  → 문서가 바뀌면 BM25도 다시 만들어야 합니다(Chroma처럼 디스크에 저장되지 않음, 메모리에만 있음).

---

## 6. Query Rewriting (질문 재작성)

### 6-1. 해결하려는 문제

```
질문 : "몸이 아파서 쉬어야 하면 어떻게 해요?"
토큰 : ['몸', '아프', '쉬', '하', '어떻', '하']
문서 : "... 병가는 ... 진단서 ..."
```

질문에 **"병가"라는 단어가 없으니 BM25는 못 찾고**, 벡터 검색도 "아파서 쉰다"와 "병가 규정" 사이의 거리가 생각보다 멀어서 실패했습니다.
그래서 검색하기 전에 LLM에게 **"문서에 실제로 쓰였을 법한 용어로 검색어를 바꿔 달라"** 고 시킵니다.

### 6-2. 재작성 프롬프트 뜯어보기

```python
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "너는 사내 업무 가이드 검색어 생성기야.\n"
     "가이드 목차: 회사와 업무 원칙, 근무 방식, ..., 휴가·비용·복지, FAQ와 연락처\n"   # ①
     "사용자 질문을 가이드 문서에 실제로 쓰였을 법한 공식 용어로 바꾼 검색어 한 줄로 만들어.\n"  # ②
     "- 일상 표현은 사내 용어로 바꿔 (예: 쓴 돈 돌려받기 → 비용 정산, ...)\n"          # ③
     "- 질문에 있는 고유명사·숫자·채널명은 그대로 유지해\n"                             # ④
     "- 설명 없이 검색어만 출력해"),                                                   # ⑤
    ("human", "{question}"),
])
```

| 번호 | 왜 넣었나 |
| --- | --- |
| ① 목차 | LLM은 이 문서를 본 적이 없습니다. 목차를 주면 "어떤 용어 체계를 쓰는 문서인지" 힌트가 됩니다. |
| ② 공식 용어 | 핵심 지시. "아파서 쉰다" → "병가" |
| ③ 예시 | 예시 1~2개를 주면(few-shot) LLM이 원하는 변환 스타일을 훨씬 잘 따라 합니다. |
| ④ 고유명사 유지 | `LearnHub`, `#help-urgent`, `15만원`을 LLM이 멋대로 바꾸면 BM25의 장점이 사라지므로 막아 둡니다. |
| ⑤ 검색어만 | "검색어는 다음과 같습니다: ..." 같은 문장이 섞이면 그 단어들까지 검색에 들어가 노이즈가 됩니다. |

실제 재작성 결과:

```
LearnHub에서는 뭘 신청하나요?        →  LearnHub 교육 과정 신청
#help-urgent 채널은 언제 쓰나요?      →  #help-urgent 채널 사용 기준 및 응답 절차
출장 숙박비 한도가 얼마예요?          →  출장비 지급기준(숙박비)
몸이 아파서 쉬어야 하면 어떻게 해요?  →  병가 신청 및 결근 신고 절차     ← "병가" 등장!
서비스가 갑자기 멈추면 누가 지휘하나요? → 서비스 장애 발생 시 사고 지휘체계 및 책임자
```

### 6-3. `rewrite_chain`

```python
rewrite_chain = rewrite_prompt | llm | StrOutputParser()
```

| 단계 | 입력 | 출력 |
| --- | --- | --- |
| `rewrite_prompt` | `{"question": "몸이 아파서..."}` | 시스템+사용자 메시지 묶음 (`ChatPromptValue`) |
| `llm` | 메시지 묶음 | `AIMessage(content="병가 신청 및 결근 신고 절차")` |
| `StrOutputParser()` | `AIMessage` | `"병가 신청 및 결근 신고 절차"` (순수 문자열) |

`StrOutputParser`가 꼭 필요한 이유: 다음에 이어질 리트리버는 **문자열**을 받기 때문입니다. `AIMessage` 객체를 그대로 넘기면 안 됩니다.

### 6-4. `rewrite_retriever` — 재작성과 검색을 한 덩어리로

```python
rewrite_retriever = (lambda q: {"question": q}) | rewrite_chain | hybrid_retriever
```

데이터가 흘러가는 모습:

```
"몸이 아파서 쉬어야 하면 어떻게 해요?"            (str)
        │  lambda q: {"question": q}
        ▼
{"question": "몸이 아파서 쉬어야 하면 어떻게 해요?"}  (dict) ← 프롬프트의 {question} 자리에 맞춤
        │  rewrite_chain (프롬프트 → LLM → 문자열)
        ▼
"병가 신청 및 결근 신고 절차"                       (str)
        │  hybrid_retriever
        ▼
[Document, Document, Document, Document]            (list[Document])
```

- **왜 lambda가 있나?** 프롬프트는 `{"question": ...}` 형태의 dict를 기대합니다. 문자열을 dict로 포장해 주는 어댑터입니다.
  (참고: 변수가 하나뿐인 프롬프트는 LangChain이 문자열을 알아서 감싸 주기도 하지만, 이렇게 명시하는 편이 읽기 쉽고 안전합니다.)
- **그냥 파이썬 lambda인데 어떻게 `|`로 연결되나?** 7-1에서 설명합니다.
- 이렇게 만든 `rewrite_retriever`도 하나의 Runnable이라서 `retriever`, `hybrid_retriever`와 **똑같이 `.invoke(q)`로 쓸 수 있습니다.**
  그래서 비교 코드에서 세 리트리버를 같은 for 문으로 돌릴 수 있었습니다.

### 6-5. 핵심 설계: "검색은 재작성 질문으로, 답변은 원래 질문으로"

재작성된 검색어("병가 신청 및 결근 신고 절차")는 **검색용으로만** 씁니다. 답변을 만드는 LLM에는 **사용자의 원래 질문**을 줍니다.

- 재작성은 정보를 잃을 수 있습니다. 예: "제주도 2박 3일 출장 숙박비 한도와 정산 기한"을 "출장비 지급기준"으로 줄이면 "정산 기한" 부분이 빠질 수 있음
- 답변은 사용자가 실제로 물은 것에 맞춰야 합니다

이 분리가 7-3의 최종 체인에서 어떻게 구현되는지 보겠습니다.

---

## 7. LCEL 체인 구성 — `|`와 dict가 실제로 하는 일

### 7-1. `|` 연산자와 자동 변환(coercion)

LCEL에서 `A | B`는 "A의 출력을 B의 입력으로 넣는 새 Runnable(`RunnableSequence`)"을 만듭니다.
`|` 양쪽 중 **하나라도 Runnable이면**, 나머지는 LangChain이 자동으로 Runnable로 바꿔 줍니다.

| 파이썬 값 | 자동으로 바뀌는 것 | 동작 |
| --- | --- | --- |
| 함수, lambda | `RunnableLambda` | 입력을 함수에 넣고 반환값을 출력 |
| dict `{키: Runnable}` | `RunnableParallel` | **같은 입력**을 각 값에 넣고, 결과를 같은 키의 dict로 모음 |

`(lambda q: {"question": q}) | rewrite_chain`에서 파이썬은 먼저 lambda의 `__or__`를 찾지만 함수에는 그런 게 없습니다.
그러면 오른쪽 `rewrite_chain`의 `__ror__`가 호출되고, 여기서 lambda를 `RunnableLambda`로 감싸서 연결합니다.
→ 그래서 **맨 앞이 lambda나 dict여도, 바로 뒤에 Runnable이 있으면 동작**합니다. (`lambda | lambda`는 둘 다 Runnable이 아니라서 에러)

### 7-2. 답변 프롬프트와 `format_docs`

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "... 아래 [문서]에 있는 내용만 근거로 한국어로 답해.\n"
               "답을 찾았으면 마지막 줄에 근거 페이지를 '(출처: N페이지)' 형식으로 적어.\n"
               "문서에 없는 내용이면 출처 없이 '가이드에서 찾을 수 없습니다'라고만 답해.\n\n"
               "[문서]\n{context}"),
    ("human", "{question}"),
])
```

변수가 `{context}`, `{question}` **두 개**입니다. 그래서 이 프롬프트에는 두 키를 가진 dict가 들어와야 합니다.

```python
def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['page'] + 1}페이지]\n{d.page_content}" for d in docs)
```

리트리버는 `list[Document]`를 주지만 프롬프트에는 **문자열**을 넣어야 합니다. 이 함수가 변환해 주며,
각 청크 앞에 `[8페이지]` 같은 머리말을 붙여서 LLM이 `(출처: 8페이지)`를 적을 수 있게 합니다. 결과 예:

```
[8페이지]
연차는 최소 1일 전 신청 ...

[9페이지]
Q. 재택근무 ...
```

### 7-3. 최종 체인 한 단계씩

```python
final_rag_chain = (
    {"context": rewrite_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

`final_rag_chain.invoke("몸이 아파서 쉬어야 하면 어떻게 해요?")`를 부르면:

```
입력: "몸이 아파서 쉬어야 하면 어떻게 해요?"
 │
 ▼ ① RunnableParallel (dict)  ── 같은 입력을 두 갈래로 동시에 보냄 ──┐
 │                                                                   │
 │   "context" 갈래                                    "question" 갈래
 │   rewrite_retriever                                 RunnablePassthrough()
 │     → 재작성: "병가 신청 및 결근 신고 절차"           → 입력을 그대로 통과
 │     → 하이브리드 검색: [Doc×4]                      → "몸이 아파서 쉬어야 하면 어떻게 해요?"
 │   | format_docs
 │     → "[3페이지]\n...\n\n[8페이지]\n..."
 │                                                                   │
 ▼ 결과를 dict로 모음 ◄──────────────────────────────────────────────┘
 {"context": "[3페이지]\n...", "question": "몸이 아파서 쉬어야 하면 어떻게 해요?"}
 │
 ▼ ② prompt        → {context}, {question} 자리를 채운 메시지
 ▼ ③ llm           → AIMessage("병가는 ... (출처: 8페이지)")
 ▼ ④ StrOutputParser → "병가는 ... (출처: 8페이지)"
```

포인트:

1. **dict가 `RunnableParallel`이 되는 부분이 핵심입니다.** 입력(원래 질문) 하나가 두 갈래로 **복사**되어 각각 처리됩니다.
2. `context` 갈래에서만 재작성이 일어납니다. `question` 갈래는 `RunnablePassthrough()`라서 **원래 질문이 그대로** 프롬프트에 들어갑니다.
   → 6-5의 "검색은 재작성, 답변은 원문"이 이 구조 덕분에 자연스럽게 구현됩니다.
3. `rewrite_retriever | format_docs`에서 `format_docs`는 일반 함수지만 앞에 Runnable이 있으니 자동으로 `RunnableLambda`가 됩니다.
4. `RunnableParallel`은 `invoke` 시 두 갈래를 스레드로 **병렬 실행**합니다(여기선 question 갈래가 즉시 끝나서 체감 차이는 없음).

첫 번째 `rag_chain`과 비교하면 **딱 한 곳만 다릅니다**:

```python
rag_chain       = {"context": retriever         | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
final_rag_chain = {"context": rewrite_retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
```

리트리버가 모두 같은 인터페이스(문자열 → `list[Document]`)를 가진 Runnable이라서, **부품만 갈아 끼우면** 체인 전체를 업그레이드할 수 있습니다. 이게 `BaseRetriever` 상속과 LCEL을 쓴 이유입니다.

### 7-4. 질문 1개당 실제 호출 횟수

| 호출 | 횟수 | 위치 |
| --- | --- | --- |
| LLM (gpt-5-mini) | **2회** | 재작성 1회 + 답변 생성 1회 |
| 임베딩 API | 1회 | 벡터 검색에서 재작성된 검색어를 임베딩 |
| BM25 | 로컬 계산 | API 비용 없음 |

### 7-5. 디버깅 팁 — 중간 결과 보기

체인이 한 줄로 묶여 있으면 어디서 틀렸는지 안 보입니다. 부품을 따로 `invoke` 하면 됩니다.

```python
q = "출장을 다녀온 후 비용 정산은 언제까지 완료해야 하나요?"
print(rewrite_chain.invoke({"question": q}))            # ① 재작성 결과가 적절한가?
for d in rewrite_retriever.invoke(q):                  # ② 정답 청크가 검색됐는가?
    print(d.metadata["page"] + 1, d.page_content[:80])
print(format_docs(rewrite_retriever.invoke(q)))         # ③ LLM이 실제로 보는 문서
```

이 방법은 부품마다 따로 호출하기 때문에 **재작성 LLM이 매번 다시 실행**됩니다. 그래서 ①에서 본 검색어와 ②에서 실제로 쓰인 검색어가 다를 수 있어요. 한 번 실행한 결과를 모두 보려면 7-6의 체인을 쓰세요.

### 7-6. 근거를 함께 반환하는 체인 — `traced_rag_chain`

`final_rag_chain`은 답변 **문자열만** 돌려줍니다. 그래서 답이 틀렸을 때 검색에서 틀렸는지 LLM이 틀렸는지 알 수 없어요.
`RunnablePassthrough.assign`으로 단계마다 결과를 dict에 **덧붙이면**, 한 번 실행해서 중간 결과를 모두 받을 수 있습니다.

```python
answer_chain = prompt | llm | StrOutputParser()

traced_rag_chain = (
    RunnablePassthrough.assign(rewritten=rewrite_chain)             # ① 재작성된 검색어
    .assign(docs=itemgetter("rewritten") | hybrid_retriever)        # ② 검색어로 찾은 청크
    .assign(answer=(lambda x: {"context": format_docs(x["docs"]),   # ③ 답변은 원래 질문으로 생성
                               "question": x["question"]})
                   | answer_chain)
)

r = traced_rag_chain.invoke({"question": q})
r["rewritten"], r["docs"], r["answer"]
```

dict가 단계를 거치며 쌓이는 모습:

```
{"question"}
  └ .assign(rewritten=…)  → {"question", "rewritten"}
     └ .assign(docs=…)    → {"question", "rewritten", "docs"}
        └ .assign(answer=…) → {"question", "rewritten", "docs", "answer"}
```

- `.assign(key=runnable)`은 **입력 dict 전체**를 runnable에 넘기고, 그 결과를 `key`로 추가합니다. 기존 키는 그대로 둡니다.
- `rewrite_chain`은 `{"question"}`만 필요한데 dict 전체를 받아도 괜찮습니다. 프롬프트는 자기가 쓰는 변수만 꺼내 씁니다.
- `itemgetter("rewritten")`으로 **재작성된 검색어**만 꺼내 리트리버에 넘깁니다. 답변 프롬프트에는 **원래 질문**을 넘깁니다(6-5와 같은 설계).
- 입력 형태가 `final_rag_chain`과 다릅니다. 문자열이 아니라 `{"question": ...}` dict를 넘겨야 해요.

**결과를 읽는 법**

| 상황 | 원인 | 손볼 곳 |
| --- | --- | --- |
| 정답 청크가 `docs`에 **없음** | 검색 실패 | 재작성 프롬프트, BM25/벡터 가중치, `k` |
| 정답 청크가 `docs`에 **있는데** 답이 틀림 | 답변 생성 실패 | 답변 프롬프트, 모델 |
| 문서에 원래 없는 내용 | 정상 | "가이드에서 찾을 수 없습니다"가 맞는 답 |

실제 예시(Q-5 재택근무): 청크가 `[(9,0), (8,392), (3,387), (8,0)]`로 나왔습니다. FAQ(9페이지)의 "이월 불가·예외 승인"은 답변에 들어갔어요.
하지만 2페이지 근무 방식 표에 있는 "매주 월요일까지 캘린더에 WFH 표시"는 **검색되지 않아서** 답변에서 빠졌습니다. 이건 LLM이 아니라 검색 쪽에서 고쳐야 하는 문제예요.

### 7-7. LangSmith 트레이싱 — 실행 과정 전체를 웹에서 보기

7-6 체인은 **최종 결과**만 dict로 보여 줍니다. LangSmith는 **모든 단계**를 트리로 기록해요. 하이브리드 검색 안의 벡터 검색, 단계별 프롬프트 원문, 토큰 수와 걸린 시간까지 볼 수 있습니다.

`.env`에 아래를 넣으면 코드 수정 없이 켜집니다. 키는 https://smith.langchain.com → Settings → API Keys에서 발급받아요.

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-one
```

노트북의 임베딩 셀 바로 뒤에 있는 "LangSmith 트레이싱" 셀이 키가 있는지 확인해서 트레이싱을 켜고 끕니다. 키가 없으면 끈 채로 넘어가서 나머지 셀은 그대로 돌아갑니다.

웹 UI에서 보이는 트리:

```
RunnableSequence (traced_rag_chain)
 ├ rewrite_chain        → ChatPromptTemplate → ChatOpenAI → 검색어
 ├ HybridRetriever      → 최종 청크 4개
 │   └ VectorStoreRetriever (MMR)   ← run_manager.get_child() 덕분에 하위 단계로 기록
 └ answer_chain         → ChatPromptTemplate → ChatOpenAI → 답변
```

- BM25 검색은 LangChain Runnable이 아니라 트리에 따로 나오지 않습니다. `HybridRetriever`의 출력으로만 보여요.
- 질문과 청크 내용이 LangSmith 서버로 전송됩니다. 실제 사내 문서라면 전송해도 되는지 먼저 확인하세요.

---

## 8. 실험 결과 요약

표기: `O/X` = 1순위 청크에 정답 키워드가 있는가, 숫자 = 4개 중 정답 키워드가 들어 있는 청크 수

| 질문 (정답 키워드) | MMR | Hybrid | Rewrite+Hybrid |
| --- | --- | --- | --- |
| LearnHub에서는 뭘 신청? (LearnHub) | X0 | X1 | **O1** |
| #help-urgent 언제? (#help-urgent) | O3 | O2 | O2 |
| 출장 숙박비 한도? (15만원) | O2 | O2 | O2 |
| 아파서 쉬려면? (병가) | X0 | X0 | **X1** |
| 서비스 멈추면 누가 지휘? (장애) | X1 | O3 | O2 |

- 정답 청크를 **하나도 못 가져온 질문**: MMR 2개 → Hybrid 1개 → **Rewrite+Hybrid 0개**
- 답변 LLM은 1순위만이 아니라 4개를 모두 읽으므로 "정답 청크를 놓치지 않는 것"이 더 중요 → 최종 체인에 Rewrite+Hybrid 채택
- 질문 5개짜리 소규모 실험이고, 재작성은 LLM 출력이라 실행마다 결과가 조금씩 달라질 수 있음

---

## 9. 알아두면 좋은 점 · 개선 아이디어

1. **해시태그가 BM25에서 빠짐** — 5-3 참고. `tokenize` 필터에 `"W"`를 추가하면 `#help-urgent`가 키워드로 잡힙니다.
2. **"찾을 수 없습니다" 답변 점검** — `traced_rag_chain`(7-6)으로 다시 실행했을 때 Q-4(노트북 분실 신고)와 Q-8(지각 보고)이 "가이드에서 찾을 수 없습니다"로 나왔습니다. 두 내용은 가이드에 명시돼 있지 않으므로 올바른 답입니다.
   반대로 Q-5(재택근무)는 2페이지 근무 방식 표를 검색하지 못해 답변이 불완전했습니다. 재작성 LLM의 출력은 실행마다 달라서 결과가 바뀔 수 있으니, 7-6 체인으로 `docs`를 함께 확인하세요.
3. **복합 질문** — "숙박비 한도와 정산 기한을 각각" 같은 질문은 검색어 하나로 두 주제를 다 찾기 어렵습니다. 질문을 여러 개로 쪼개 각각 검색하는 Multi-Query 방식이 다음 단계가 될 수 있습니다.
4. **BM25는 메모리에만 있음** — 문서를 추가·수정하면 `BM25Okapi`를 다시 만들어야 합니다. Chroma에서 지워지지 않은 옛 청크도 따로 정리가 필요합니다.
5. **가중치·rrf_k 튜닝** — 지금은 0.5/0.5, 60 고정입니다. 평가 질문을 20~50개 정도로 늘린 뒤 바꿔 가며 비교해야 제대로 된 결론을 낼 수 있습니다.
