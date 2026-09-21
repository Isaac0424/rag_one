# 2026-09-21 — LangChain Document Loaders & Text Splitters

RAG 파이프라인의 앞단(**로드 → 분할**)을 LangChain으로 실습한 기록.
"각 함수/클래스를 **언제 쓰는가**, 뭐가 좋고 뭐가 불편한가"를 비교하는 데 초점을 뒀습니다.

```
RAG 파이프라인
[1] Load ──► [2] Split ──► [3] Embed ──► [4] Store ──► [5] Retrieve ──► [6] Generate
 ▲── 이번 정리 범위 ──▲
```

---

## 0. 한눈에 고르기

```
입력이 무엇인가?
├─ 파일/웹  → Document Loader 선택 (1장)
└─ 이미 텍스트
     ├─ 구조 있음 (MD / HTML / JSON / 코드) → 구조 기준 분할 후 문자 기준 2차 분할
     ├─ 임베딩 토큰 한계가 빡빡함          → 토큰 기준 분할
     ├─ 검색 품질이 최우선 + 비용 OK       → SemanticChunker
     └─ 그 외 전부                         → RecursiveCharacterTextSplitter (기본값)
```

종류를 가로지르는 비교표와 현업 적용 방식은 [4장](#4-chunker-전체-비교).

---

## 1. Document 객체

모든 로더의 출력 단위는 `langchain_core.documents.Document`이며 두 개의 필드를 가집니다.

| 필드 | 설명 |
| --- | --- |
| `page_content` | 실제 텍스트 본문 |
| `metadata` | `source`, `page`, `author` 등 임의의 키/값 (나중에 검색 필터링에 사용) |

```python
from langchain_core.documents import Document

document = Document(page_content="안녕하세요? 이게 객체형태의 도큐먼트 입니다.")
document.metadata["source"] = "TeddyNote"
document.metadata["page"] = 1
```

→ [document_object.ipynb](document_object.ipynb)

---

## 2. Document Loaders 비교

로더는 모두 위 `Document`의 리스트를 돌려줍니다.

| 로더 | 장점 | 단점 / 주의 | 이럴 때 쓴다 |
| --- | --- | --- | --- |
| `PyPDFLoader` | 페이지당 Document 1개, `metadata["page"]` 자동. 설치 가볍고 빠름 | 표·다단 레이아웃이 깨짐. 스캔 PDF(이미지)는 텍스트가 안 나옴 | 텍스트 위주 PDF 보고서 |
| `CSVLoader` | 행 1개 = Document 1개라 행 단위 검색에 딱 맞음. `csv_args`로 구분자·`fieldnames` 제어 | `page_content`가 `컬럼: 값` 나열이라 표 구조가 사라짐 | 행 자체가 검색 단위일 때 (고객 목록, 로그) |
| `UnstructuredCSVLoader` | `mode="elements"` 시 `metadata["text_as_html"]`로 **표 구조 보존** | 전체가 Document 1개로 뭉침. 의존성 무거움 | 표 모양 그대로 LLM에 보여줘야 할 때 |
| `DataFrameLoader` | pandas 전처리(필터·결측 처리) 후 로드 가능. `page_content_column`으로 본문 컬럼 선택, 나머지는 자동 메타데이터 | DataFrame을 메모리에 먼저 올려야 함 | 정제가 필요한 표 데이터 |
| `DirectoryLoader` | `glob` 패턴으로 폴더 통째 로드. `loader_cls`로 파일 유형별 로더 지정 | 기본 Unstructured가 **인코딩을 추측** → 한글 Windows에서 CP949로 깨짐. `libmagic` 필요 | 문서 폴더 일괄 적재 |
| `WebBaseLoader` | `bs4.SoupStrainer`로 **필요한 영역만** 파싱 → 광고/네비 제거. 비동기·프록시 지원 | 사이트마다 CSS 클래스가 달라 선택자 수작업. JS 렌더링 페이지는 못 읽음 | 뉴스·블로그 등 정적 페이지 |
| `HWPLoader` (직접 구현) | 국내 문서 대응. `BaseLoader` 상속으로 다른 로더와 동일 인터페이스 | `olefile` + `zlib`로 직접 파싱 → 서식·표 유실, 버전에 따라 실패 가능 | 관공서 HWP 문서 |

### 로드 메서드 4종

| 메서드 | 장점 | 단점 | 상황 |
| --- | --- | --- | --- |
| `load()` | 가장 단순, 바로 리스트 | 전부 메모리에 올림 | 파일 몇 개, 탐색 단계 |
| `lazy_load()` | 제너레이터라 메모리 일정 | 길이(`len`)를 미리 못 봄 | 대용량 / 스트리밍 적재 |
| `aload()` | 여러 URL 동시 요청으로 빠름 | Jupyter에서 이벤트 루프 충돌 → `nest_asyncio.apply()` 필요 | 웹 다중 크롤링 |
| `load_and_split(text_splitter=...)` | 로드 + 분할 한 번에 | 분할 전 원본 Document를 못 봄 | 파이프라인이 이미 확정됐을 때 |

### CSVLoader 행 → XML 변환

`page_content`가 `컬럼: 값` 나열이라 LLM이 구조를 놓치기 쉬워, 태그로 감싸면 인식률이 올라갑니다.

```python
for doc in docs:
    row_str = "<row>"
    for element in doc.page_content.split("\n"):
        splitted = element.split(":")
        col, value = ":".join(splitted[:-1]), splitted[-1]
        row_str += f"<{col}>{value.strip()}</{col}>"   # 닫는 '>' 빠뜨리기 쉬움
    row_str += "</row>"
```

→ [csv_loader.ipynb](csv_loader.ipynb) · [directory_loader.ipynb](directory_loader.ipynb) · [webbase_loader.ipynb](webbase_loader.ipynb) · [hwp_loader.ipynb](hwp_loader.ipynb)

---

## 3. Text Splitters 비교

### 3-1. 문자 기준

| 스플리터 | 분할 기준 | 장점 | 단점 | 상황 |
| --- | --- | --- | --- | --- |
| `CharacterTextSplitter` | `separator` **하나** (기본 `"\n\n"`) | 동작이 예측 가능, 구분자가 명확하면 깔끔 | 구분자가 없으면 **`chunk_size`를 넘겨버림** (경고만 뜨고 안 잘림) | 문단 구분이 확실한 문서 |
| `RecursiveCharacterTextSplitter` | `["\n\n", "\n", " ", ""]` **순차 시도** | 문단 → 줄 → 단어로 물러나며 자르니 크기 보장 + 의미가 덜 깨짐 | 구분자 우선순위가 한국어에 최적은 아님 | **잘 모르겠으면 이것** (범용 기본값) |

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,      # 청크 최대 크기 (length_function 기준)
    chunk_overlap=50,    # 경계에서 잘린 문맥 보험
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([file])  # → List[Document] (메타데이터 유지)
texts = text_splitter.split_text(file)          # → List[str]      (문자열만)
```

`create_documents([a, b], metadatas=[{...}, {...}])` — 원문별 메타데이터 부착.

**`create_documents` vs `split_text`**: 벡터스토어에 넣을 거면 메타데이터가 붙는 `create_documents`, 중간 확인·가공용이면 `split_text`.

→ [character_text_splitter.ipynb](character_text_splitter.ipynb) · [recursive_character_splitter.ipynb](recursive_character_splitter.ipynb)

### 3-2. 토큰 기준

임베딩 모델의 한계는 **문자 수가 아니라 토큰 수**입니다. 한글은 문자당 토큰이 많아 문자 기준으로 맞추면 초과하기 쉽습니다.

| 방식 | 장점 | 단점 | 상황 |
| --- | --- | --- | --- |
| `CharacterTextSplitter.from_tiktoken_encoder(...)` | OpenAI 토크나이저와 **정확히 일치** → 한계 초과 없음 | OpenAI 모델 전용 | OpenAI 임베딩/LLM 사용 시 |
| `SentenceTransformersTokenTextSplitter` | 실제 사용할 임베딩 모델(`all-MiniLM-L6-v2` 등) 토크나이저 기준 | 모델 로딩이 무겁고 느림 | 로컬 오픈소스 임베딩 |
| `SpacyTextSplitter` | 문장 경계를 문법적으로 정확히 | `en_core_web_sm` 다운로드 필요, 느림. 한국어는 별도 모델 | 문장 단위가 중요한 영문 |
| `NLTKTextSplitter` | 가볍고 빠른 문장 분리 | `punkt_tab` 다운로드 필요. 한국어 정확도 낮음 | 간단한 영문 문장 분리 |

```python
count_start_and_stop_tokens = 2   # [CLS], [SEP] 같은 특수 토큰
text_token_count = splitter.count_tokens(text=file) - count_start_and_stop_tokens
```

→ [token_text_splitter.ipynb](token_text_splitter.ipynb)

### 3-3. 구조 기준

| 스플리터 | 장점 | 단점 | 상황 |
| --- | --- | --- | --- |
| `MarkdownHeaderTextSplitter` | 헤더 계층을 **메타데이터로 보존** → 검색 결과에 "어느 섹션인지"가 남음. `strip_headers=False`면 본문에도 헤더 유지 | 크기 보장 안 됨 (섹션이 길면 거대 청크) | 문서 / 위키 / 기술 문서 |
| `HTMLHeaderTextSplitter` | `h1`~`h3` 계층을 메타데이터로 | 태그가 엉망인 실제 웹페이지에선 불안정. 크기 보장 없음 | 구조가 정돈된 HTML |
| `RecursiveJsonSplitter` | 중첩 객체 구조를 **깨지 않고** 분할. 각 청크가 유효한 JSON | `max_chunk_size`보다 작으면 **아예 안 쪼갬(청크 1개)**. 리스트를 잘 못 다룸 | API 스펙, 설정 파일, 중첩 JSON |
| `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` | `def` / `class` 등 언어 문법 구분자 사용 → 함수가 중간에 안 잘림 | 언어별 지원 범위 한정 | 소스코드 인덱싱 |

**핵심 패턴 — 2단 분할**: 구조 기준으로 1차 → 문자 기준으로 2차. 메타데이터를 살리면서 크기까지 맞출 수 있습니다.

```python
md_header_splits = markdown_splitter.split_text(markdown_document)
splits = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20
).split_documents(md_header_splits)   # split_documents: 메타데이터 승계
```

`RecursiveJsonSplitter`는 리스트가 있으면 경계가 잘 안 잡혀 `convert_lists=True`로 리스트를 인덱스 키 객체(`{"0": ..., "1": ...}`)로 바꿔 분할합니다.

→ [html_header_splitter.ipynb](html_header_splitter.ipynb) · [recursive_json_splitter.ipynb](recursive_json_splitter.ipynb) · [semantic_chunker.ipynb](semantic_chunker.ipynb)

### 3-4. 의미 기준 — SemanticChunker

문장 임베딩 간 거리를 계산해 **의미가 바뀌는 지점**에서 자릅니다.

- 장점: 주제 단위로 잘려 검색 품질이 가장 좋음. 문장 중간이 안 잘림
- 단점: **임베딩 API 호출 비용·시간** 발생(`OPENAI_API_KEY` 필요), 청크 크기가 들쭉날쭉해 토큰 한계 보장 없음, 재현성이 낮음
- 상황: 문단 구분이 없는 긴 산문, 검색 품질이 비용보다 중요한 프로덕션

```python
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
```

| `breakpoint_threshold_type` | 기준 | 실습값 | 성향 |
| --- | --- | --- | --- |
| `percentile` (기본) | 거리 상위 N% 지점에서 분할 | 70 | 직관적, 값을 낮출수록 청크 수 증가 |
| `standard_deviation` | 평균 + N×표준편차 초과 | 1.25 | 이상치성 전환점만 |
| `interquartile` | IQR × N 초과 | 0.5 | 분포에 덜 민감 |

→ [semantic_chunker.ipynb](semantic_chunker.ipynb)

---

## 4. Chunker 전체 비교

3장은 분류 **안에서** 비교했고, 여기서는 분류를 **가로질러** 봅니다.

### 4-1. 한 표로 보는 전 종류

| Chunker | 경계 기준 | 크기 보장 | 의미 보존 | 속도 | 비용 | 메타데이터 | 결정적\* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CharacterTextSplitter` | 구분자 1개 | ✗ (초과 가능) | 하 | 매우 빠름 | 없음 | ✗ | ✓ |
| `RecursiveCharacterTextSplitter` | 구분자 우선순위 | ✓ | 중 | 매우 빠름 | 없음 | ✗ | ✓ |
| `...from_tiktoken_encoder` | 토큰 수 | ✓ (토큰 기준) | 중 | 빠름 | 없음 | ✗ | ✓ |
| `SentenceTransformersTokenTextSplitter` | 모델 토큰 수 | ✓ (토큰 기준) | 중 | 느림 | 로컬 연산 | ✗ | ✓ |
| `SpacyTextSplitter` / `NLTKTextSplitter` | 문장 | 대략 | 중상 | 느림 / 보통 | 없음 | ✗ | ✓ |
| `MarkdownHeaderTextSplitter` | 헤더 계층 | ✗ | 상 | 매우 빠름 | 없음 | **✓ 헤더** | ✓ |
| `HTMLHeaderTextSplitter` | 헤더 계층 | ✗ | 상 | 빠름 | 없음 | **✓ 헤더** | ✓ |
| `RecursiveJsonSplitter` | JSON 노드 | ✓ (근사) | 상 | 빠름 | 없음 | ✗ | ✓ |
| `...from_language(...)` | 언어 문법 | ✓ | 상 | 매우 빠름 | 없음 | ✗ | ✓ |
| `SemanticChunker` | 임베딩 거리 | **✗** | **최상** | 매우 느림 | **API 과금** | ✗ | ✗ |

\* 결정적 = 같은 입력에 항상 같은 출력. `SemanticChunker`는 임베딩 모델/버전에 따라 결과가 달라집니다.

**표에서 읽히는 트레이드오프**: 의미 보존이 올라갈수록 크기 보장이 무너지고 비용이 붙습니다. 그래서 현업에서는 **의미 기준으로 1차, 크기 기준으로 2차**를 겹쳐 쓰는 게 정석입니다.

### 4-2. 선택 기준 3가지

| 질문 | 예 → | 아니오 → |
| --- | --- | --- |
| 문서에 **구조**(헤더/키/함수)가 있는가? | 구조 기준 1차 + 문자 기준 2차 | 문자 기준 단독 |
| 임베딩 **토큰 한계**가 빡빡한가? | 토큰 기준(`from_tiktoken_encoder`) | 문자 기준으로 충분 |
| 검색 품질을 위해 **비용·시간**을 쓸 수 있는가? | `SemanticChunker` 검토 | `RecursiveCharacterTextSplitter` |

### 4-3. 현업 적용 방식

**① 기본 레시피 — 구조 1차 + 크기 2차**

가장 많이 쓰이는 조합입니다. 헤더 메타데이터는 남기고, 크기는 임베딩 모델에 맞춥니다.

```python
# 1차: 구조로 나누고 헤더를 메타데이터에 남김
md_splits = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
).split_text(markdown_document)

# 2차: 토큰 기준으로 크기를 맞춤 (메타데이터 승계)
splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
).split_documents(md_splits)
```

`split_documents`가 1차의 메타데이터를 그대로 물려주는 게 핵심입니다. 나중에 "3장 섹션에서만 검색" 같은 필터가 가능해집니다.

**② 문서 유형별 실전 조합**

| 문서 유형 | 조합 | 이유 |
| --- | --- | --- |
| 사내 위키 / 기술 문서 (MD) | `MarkdownHeaderTextSplitter` → `Recursive...from_tiktoken_encoder` | 섹션 출처가 답변 근거로 쓰임 |
| PDF 보고서 | `PyPDFLoader` → `Recursive...from_tiktoken_encoder` | 페이지 메타데이터가 이미 있음. 표는 별도 파이프라인 권장 |
| 웹 크롤링 | `WebBaseLoader`(SoupStrainer) → `HTMLHeaderTextSplitter` → 문자 기준 | 로드 단계에서 노이즈를 먼저 제거하는 게 더 효과적 |
| API 스펙 / 설정 JSON | `RecursiveJsonSplitter(convert_lists=True)` | 청크가 유효한 JSON이라 LLM이 구조를 이해함 |
| 소스코드 | `Recursive...from_language(Language.PYTHON)` | 함수/클래스가 중간에 안 잘림 |
| CSV / 표 데이터 | `CSVLoader` 또는 `DataFrameLoader` (분할 안 함) | 행 자체가 이미 적절한 청크. 억지로 쪼개면 의미 손상 |
| 회의록·인터뷰 등 긴 산문 | `SemanticChunker` → 크기 상한 재분할 | 문단 구분이 없어 구조 기준을 못 씀 |

**③ 파라미터 실전 감각**

| 파라미터 | 실무 기본값 | 근거 |
| --- | --- | --- |
| `chunk_size` | 300~800 토큰 | 작으면 문맥 부족, 크면 노이즈가 섞여 검색 정밀도 하락 |
| `chunk_overlap` | `chunk_size`의 10~20% | 경계에서 잘린 문장 보험. 너무 키우면 중복 검색·저장 비용 증가 |
| 길이 단위 | **토큰** (`from_tiktoken_encoder`) | 한글은 문자당 토큰이 많아 문자 기준이 위험 |

**④ SemanticChunker를 현업에서 쓸 때 주의**

- 문서 수 × 문장 수만큼 임베딩 호출 → **인덱싱 비용이 선형 증가**. 대량 적재 시 부담
- 청크 크기 상한이 없어 임베딩 모델 한계를 넘을 수 있음 → 뒤에 크기 기준 재분할을 반드시 붙일 것
- 결과가 비결정적이라 **재현·A/B 테스트가 어려움**
- 현실적 절충: 전체에 쓰지 말고 **검색 품질이 안 나오는 특정 문서군에만** 적용

**⑤ 평가 없이 고르지 않기**

청킹 전략은 "이론상 좋은 것"보다 **내 데이터에서 잘 되는 것**이 정답입니다. 실무 순서:

1. `RecursiveCharacterTextSplitter`(토큰 기준)로 베이스라인 구축
2. 실제 질문 20~30개로 검색 결과를 눈으로 확인
3. 틀리는 케이스의 원인을 보고(문맥 부족 → `chunk_size`↑, 노이즈 → `chunk_size`↓, 구조 손실 → 구조 기준 도입) 한 번에 하나씩 바꾸기

---

## 5. 오늘 막혔던 것과 원인

### Q1. 노트북 열자마자 `JSONDecodeError: Expecting value: line 1 column 1`

[recursive_json_splitter.ipynb](recursive_json_splitter.ipynb) 파일이 **0바이트**였습니다. 코드 문제가 아니라 노트북 파일 자체가 비어 JSON 파싱이 실패한 것.
→ 빈 노트북 골격(`{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}`)을 넣거나 새로 생성.

### Q2. `print(texts[1])`에서 `IndexError: list index out of range`

```python
# cell 5
json_data = json.loads(texts[2])   # ← 원본을 226자짜리 청크로 덮어씀

# cell 7
texts = splitter.split_text(json_data=json_data, convert_lists=True)
print(texts[1])                    # IndexError
```

원인은 **변수 덮어쓰기**. `splitter`의 `max_chunk_size=300`인데 `json_data`가 226자로 바뀌어 있어 **쪼갤 게 없으니 청크가 1개**만 나옵니다. `texts[0]`은 있고 `texts[1]`은 없음.

`convert_lists=True`는 이 에러와 무관 — 리스트를 dict로 바꿔 더 잘게 나눌 여지를 줄 뿐, 입력이 `max_chunk_size`보다 작으면 여전히 1개입니다.

해결: 변수명을 분리하거나 cell 0을 다시 실행해 원본 복구.

```python
sub_data = json.loads(texts[2])                 # 원본 json_data 보존
print(len(texts))                               # 디버깅 1순위
texts = splitter.split_text(json_data=json_data, convert_lists=True)
```

> **교훈**: 노트북은 셀을 순서 없이 돌리다 변수가 조용히 바뀝니다. 인덱스 에러가 나면 `texts[1]`을 의심하기 전에 `len(texts)`와 **입력 변수의 현재 값**부터 확인.

### 그 외

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `DirectoryLoader` + `TextLoader` `0x80` 디코딩 에러 | Windows 한글 기본 인코딩이 CP949인데 `.md`는 UTF-8 | `loader_kwargs={"encoding": "utf-8"}` |
| 텍스트 읽기 글자 깨짐 | 위와 동일 | `open(..., encoding="utf-8")` 명시 |
| `WebBaseLoader` SSL 인증서 에러 | 사이트 인증서 검증 실패 | `loader.requests_kwargs = {"verify": True}`. 검증을 끄기 전에 원인부터 확인 |
| 노트북에서 `aload()` 이벤트 루프 충돌 | Jupyter가 이미 루프 실행 중 | `nest_asyncio.apply()` + `loader.requests_per_second` |
| `DirectoryLoader` 파일 타입 감지 실패 | `libmagic` 미설치 | `uv pip install python-magic-bin` (Windows) |
| `.md` 로드 시 의존성 에러 | 마크다운 파서 미설치 | `uv pip install "unstructured[md]"` |

---

## 6. 정리하며 얻은 감각

- **기본은 `RecursiveCharacterTextSplitter`** — 구조 정보가 필요할 때만 다른 걸 얹는다.
- **`chunk_size`는 토큰으로 생각한다** — 문자 기준은 한글에서 특히 위험.
- **`chunk_overlap`은 문맥 보험** — 경계에서 잘린 문장이 양쪽에 남도록. 보통 `chunk_size`의 10~20%.
- **메타데이터는 나중에 검색 필터가 된다** — 분할 단계에서 출처·페이지·헤더를 최대한 남긴다.
- **스플리터가 "안 쪼갠다"면 대부분 입력이 `chunk_size`보다 작은 것** — 에러 메시지보다 입력 크기를 먼저 본다.

---

## 실행 환경

```bash
uv pip install langchain langchain-community langchain-text-splitters langchain-experimental \
               langchain-openai pypdf beautifulsoup4 pandas olefile nest-asyncio \
               "unstructured[md]" python-magic-bin
```

`SemanticChunker` 실습에는 `.env`의 `OPENAI_API_KEY`가 필요합니다.

## 파일

| 노트북 | 내용 |
| --- | --- |
| [document_object.ipynb](document_object.ipynb) | Document 객체, PyPDFLoader, load/lazy_load/aload |
| [csv_loader.ipynb](csv_loader.ipynb) | CSVLoader, UnstructuredCSVLoader, DataFrameLoader, XML 변환 |
| [directory_loader.ipynb](directory_loader.ipynb) | DirectoryLoader, 인코딩 문제 해결 |
| [webbase_loader.ipynb](webbase_loader.ipynb) | WebBaseLoader, SSL·비동기·프록시 |
| [hwp_loader.ipynb](hwp_loader.ipynb) | BaseLoader 상속 커스텀 HWP 로더 |
| [character_text_splitter.ipynb](character_text_splitter.ipynb) | CharacterTextSplitter, metadatas |
| [recursive_character_splitter.ipynb](recursive_character_splitter.ipynb) | RecursiveCharacterTextSplitter |
| [token_text_splitter.ipynb](token_text_splitter.ipynb) | tiktoken, spaCy, NLTK, SentenceTransformers |
| [html_header_splitter.ipynb](html_header_splitter.ipynb) | HTMLHeaderTextSplitter |
| [recursive_json_splitter.ipynb](recursive_json_splitter.ipynb) | RecursiveJsonSplitter, convert_lists |
| [semantic_chunker.ipynb](semantic_chunker.ipynb) | SemanticChunker, Markdown 헤더, 코드 스플리터 |

### 데이터

`data/` — SPRI AI Brief PDF, titanic.csv, appendix-keywords.txt (UTF-8 / CP949), 디지털 정부혁신 추진계획.hwp, markdown/ 샘플 문서
