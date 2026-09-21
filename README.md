# RAG 학습 정리 — LangChain Document Loaders & Text Splitters

RAG 파이프라인의 앞단(**로드 → 분할**)을 LangChain으로 실습한 기록입니다.

```
RAG 파이프라인
[1] Load ──► [2] Split ──► [3] Embed ──► [4] Store ──► [5] Retrieve ──► [6] Generate
 ▲── 이번 정리 범위 ──▲
```

## 1. Document 객체

모든 로더의 출력 단위는 `langchain_core.documents.Document` 이며 두 개의 필드를 가집니다.

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

로더 공통 인터페이스:

| 메서드 | 동작 |
| --- | --- |
| `load()` | 전체를 한 번에 메모리로 로드 |
| `lazy_load()` | 제너레이터로 하나씩 로드 (대용량 파일에 적합) |
| `aload()` | 비동기 로드 (`await` 필요) |
| `load_and_split(text_splitter=...)` | 로드와 분할을 한 번에 |

→ [document_object.ipynb](2026-09-21/document_object.ipynb)

## 2. Document Loaders

| 로더 | 대상 | 핵심 포인트 |
| --- | --- | --- |
| `PyPDFLoader` | PDF | 페이지 1개당 Document 1개, `metadata["page"]` 자동 부여 |
| `CSVLoader` | CSV | 행 1개당 Document 1개. `csv_args`로 구분자·따옴표·`fieldnames` 제어, `source_column`으로 출처 컬럼 지정 |
| `UnstructuredCSVLoader` | CSV | `mode="elements"` 일 때 `metadata["text_as_html"]`로 표 구조 보존 |
| `DataFrameLoader` | pandas | `page_content_column`으로 본문 컬럼을 고르고 나머지 컬럼은 메타데이터로 |
| `DirectoryLoader` | 폴더 | `glob="**/*.md"` 패턴 매칭, `loader_cls`로 파일 유형별 로더 지정 |
| `PythonLoader` | 코드 | `DirectoryLoader`의 `loader_cls`로 넣어 소스 파일 파싱 |
| `WebBaseLoader` | 웹페이지 | `bs4.SoupStrainer`로 필요한 영역만 파싱, `header_template`으로 User-Agent 지정 |
| `HWPLoader` (직접 구현) | HWP | `olefile` + `zlib`로 본문 스트림을 해제해 파싱, `BaseLoader` 상속 |

### CSV 행을 XML 태그로 변환

`CSVLoader`의 `page_content`는 `컬럼명: 값`이 줄바꿈으로 이어진 형태라, 구조를 살려 LLM에 넣으려면 태그로 변환하는 편이 낫습니다.

```python
for doc in docs:
    row_str = "<row>"
    for element in doc.page_content.split("\n"):
        splitted = element.split(":")
        col, value = ":".join(splitted[:-1]), splitted[-1]
        row_str += f"<{col}>{value.strip()}</{col}>"
    row_str += "</row>"
```

> 처음에 닫는 태그를 `</{col}` 로 써서 `>`가 빠졌던 부분을 수정했습니다.

→ [csv_loader.ipynb](2026-09-21/csv_loader.ipynb) · [directory_loader.ipynb](2026-09-21/directory_loader.ipynb) · [webbase_loader.ipynb](2026-09-21/webbase_loader.ipynb) · [hwp_loader.ipynb](2026-09-21/hwp_loader.ipynb)

## 3. Text Splitters

### 3-1. 문자 기준

| 스플리터 | 분할 기준 | 특징 |
| --- | --- | --- |
| `CharacterTextSplitter` | `separator` 하나 (기본 `"\n\n"`) | 단순하지만 구분자가 없으면 `chunk_size`를 넘겨버림 |
| `RecursiveCharacterTextSplitter` | `["\n\n", "\n", " ", ""]` 순차 시도 | **일반 텍스트 기본값**. 문단 → 줄 → 단어 순으로 물러나며 자름 |

공통 파라미터: `chunk_size`(청크 최대 크기), `chunk_overlap`(청크 간 중복 — 문맥 단절 방지), `length_function`, `is_separator_regex`.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([file])   # → List[Document]
texts = text_splitter.split_text(file)           # → List[str]
```

`create_documents([a, b], metadatas=[{...}, {...}])` 로 원문별 메타데이터를 붙일 수 있습니다.

→ [character_text_splitter.ipynb](2026-09-21/character_text_splitter.ipynb) · [recursive_character_splitter.ipynb](2026-09-21/recursive_character_splitter.ipynb)

### 3-2. 토큰 기준

임베딩 모델의 입력 한계는 **문자 수가 아니라 토큰 수**라, 실제로는 토큰 기준 분할이 더 안전합니다.

| 방식 | 사용처 |
| --- | --- |
| `CharacterTextSplitter.from_tiktoken_encoder(...)` | OpenAI 모델 토크나이저 기준 |
| `SentenceTransformersTokenTextSplitter` | 특정 임베딩 모델(`all-MiniLM-L6-v2` 등) 토크나이저 기준 |
| `SpacyTextSplitter` | spaCy 문장 분리 (`en_core_web_sm` 다운로드 필요) |
| `NLTKTextSplitter` | NLTK 문장 분리 (`nltk.download("punkt_tab")` 필요) |

`SentenceTransformersTokenTextSplitter.count_tokens()`는 시작/종료 토큰 2개를 포함하므로, 순수 본문 토큰 수를 보려면 2를 빼야 합니다.

→ [token_text_splitter.ipynb](2026-09-21/token_text_splitter.ipynb)

### 3-3. 구조 기준

| 스플리터 | 대상 | 특징 |
| --- | --- | --- |
| `MarkdownHeaderTextSplitter` | Markdown | 헤더 레벨을 메타데이터로 보존. `strip_headers=False`면 본문에 헤더도 남김 |
| `HTMLHeaderTextSplitter` | HTML | `h1`~`h3` 기준 분할, 헤더 계층을 메타데이터로 |
| `RecursiveJsonSplitter` | JSON | 중첩 객체 구조를 유지하며 분할, `max_chunk_size` 지정 |
| `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` | 소스코드 | 언어별 문법 구분자(`def`, `class` 등) 사용 |

구조 기준으로 1차 분할한 뒤 `RecursiveCharacterTextSplitter`로 2차 분할하면, 메타데이터를 살린 채 크기까지 맞출 수 있습니다.

```python
md_header_splits = markdown_splitter.split_text(markdown_document)
splits = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20
).split_documents(md_header_splits)
```

`RecursiveJsonSplitter`는 리스트가 들어 있으면 청크 경계가 잘 잡히지 않아, `split_text(json_data=..., convert_lists=True)`로 리스트를 인덱스 키를 가진 객체로 바꿔 분할했습니다.

→ [semantic_chunker.ipynb](2026-09-21/semantic_chunker.ipynb) · [html_header_splitter.ipynb](2026-09-21/html_header_splitter.ipynb) · [recursive_json_splitter.ipynb](2026-09-21/recursive_json_splitter.ipynb)

### 3-4. 의미 기준 (SemanticChunker)

문장 임베딩 간 거리를 계산해 **의미가 바뀌는 지점**에서 자릅니다. 임베딩 API 호출이 필요합니다(`.env`의 `OPENAI_API_KEY`).

```python
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
```

| `breakpoint_threshold_type` | 기준 | 실습값 |
| --- | --- | --- |
| `percentile` (기본) | 거리 상위 N% 지점에서 분할 | 70 |
| `standard_deviation` | 평균 + N×표준편차 초과 지점 | 1.25 |
| `interquartile` | IQR × N 초과 지점 | 0.5 |

임계값을 낮출수록 분할 지점이 많아져 청크 수가 늘어납니다.

→ [semantic_chunker.ipynb](2026-09-21/semantic_chunker.ipynb)

## 4. 막혔던 부분과 해결

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `DirectoryLoader` + `TextLoader`에서 `0x80` 디코딩 에러 | `TextLoader`가 파이썬 기본 인코딩을 쓰는데 Windows 한글 환경 기본값이 CP949. `.md` 파일은 UTF-8 | `loader_kwargs={"encoding": "utf-8"}` 전달 |
| 텍스트 파일 읽기 시 글자 깨짐 | 위와 동일 | `open(..., encoding="utf-8")` 명시 |
| `WebBaseLoader` SSL 인증서 에러 | 사이트 인증서 검증 실패 | `loader.requests_kwargs = {"verify": True}` 로 명시 설정. 검증을 끄는 대신 인증서 문제 원인을 먼저 확인하는 쪽이 안전 |
| 노트북에서 `aload()` 사용 시 이벤트 루프 충돌 | Jupyter가 이미 이벤트 루프를 돌리는 중 | `nest_asyncio.apply()` 후 `loader.requests_per_second`로 요청 속도 제한 |
| `DirectoryLoader` 파일 타입 감지 실패 | `libmagic` 미설치 | `uv pip install python-magic-bin` (Windows) |
| `.md` 로드 시 Unstructured 의존성 에러 | 마크다운 파서 미설치 | `uv pip install "unstructured[md]"` |
| XML 변환 결과의 태그가 안 닫힘 | f-string에서 `>` 누락 | `f"</{col}>"` 로 수정 |

## 5. 정리하며 얻은 감각

- **어떤 스플리터를 쓸까**: 일반 텍스트는 `RecursiveCharacterTextSplitter`, 구조가 있는 문서(MD/HTML/JSON/코드)는 구조 기준으로 먼저 나누고 문자 기준으로 다시 나누기, 품질이 중요하고 비용을 감수할 수 있으면 `SemanticChunker`.
- **`chunk_size`는 토큰 기준으로 생각하기**: 문자 수로 맞추면 임베딩 모델의 토큰 한계를 넘길 수 있음.
- **`chunk_overlap`은 문맥 보험**: 경계에서 잘린 문장이 양쪽 청크에 모두 남도록.
- **메타데이터는 나중에 검색 필터가 됨**: 분할 단계에서 출처·페이지·헤더를 최대한 남겨둘 것.

## 실행 환경

```bash
uv pip install langchain langchain-community langchain-text-splitters langchain-experimental \
               langchain-openai pypdf beautifulsoup4 pandas olefile nest-asyncio \
               "unstructured[md]" python-magic-bin
```

`SemanticChunker` 실습에는 `.env`의 `OPENAI_API_KEY`가 필요합니다 (`.env`는 `.gitignore` 처리됨).

### 파일

| 노트북 | 내용 |
| --- | --- |
| [document_object.ipynb](2026-09-21/document_object.ipynb) | Document 객체, PyPDFLoader, load/lazy_load/aload |
| [csv_loader.ipynb](2026-09-21/csv_loader.ipynb) | CSVLoader, UnstructuredCSVLoader, DataFrameLoader |
| [directory_loader.ipynb](2026-09-21/directory_loader.ipynb) | DirectoryLoader, 인코딩 문제 해결 |
| [webbase_loader.ipynb](2026-09-21/webbase_loader.ipynb) | WebBaseLoader, SSL·비동기·프록시 |
| [hwp_loader.ipynb](2026-09-21/hwp_loader.ipynb) | BaseLoader 상속 커스텀 HWP 로더 |
| [character_text_splitter.ipynb](2026-09-21/character_text_splitter.ipynb) | CharacterTextSplitter, metadatas |
| [recursive_character_splitter.ipynb](2026-09-21/recursive_character_splitter.ipynb) | RecursiveCharacterTextSplitter |
| [token_text_splitter.ipynb](2026-09-21/token_text_splitter.ipynb) | tiktoken, spaCy, NLTK, SentenceTransformers |
| [html_header_splitter.ipynb](2026-09-21/html_header_splitter.ipynb) | HTMLHeaderTextSplitter |
| [recursive_json_splitter.ipynb](2026-09-21/recursive_json_splitter.ipynb) | RecursiveJsonSplitter, convert_lists |
| [semantic_chunker.ipynb](2026-09-21/semantic_chunker.ipynb) | SemanticChunker, Markdown 헤더, 코드 스플리터 |

### 데이터

`2026-09-21/data/` — SPRI AI Brief PDF, titanic.csv, appendix-keywords.txt (UTF-8 / CP949), 디지털 정부혁신 추진계획.hwp, markdown/ 샘플 문서
