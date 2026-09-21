# 문서 로더(Document Loader)

문서 로더는 다양한 형식의 원본 데이터를 읽어 LangChain의 `Document` 객체로 변환합니다.
모든 로더는 `load()` 메서드를 제공하며, 결과는 `Document`의 리스트입니다.

## Document 객체 구조

`Document`는 두 개의 필드로 이루어집니다.

- `page_content`: 문서의 본문 텍스트(문자열)
- `metadata`: 출처, 페이지 번호 등 부가 정보(딕셔너리)

## 주요 로더

### CSVLoader

CSV 파일의 각 행을 하나의 `Document`로 만듭니다. `source_column`으로 어떤 컬럼을
출처로 쓸지 지정할 수 있고, `csv_args`로 구분자나 필드명을 바꿀 수 있습니다.

### WebBaseLoader

웹페이지를 가져옵니다. `bs_kwargs`에 BeautifulSoup의 `SoupStrainer`를 넘기면
필요한 영역만 골라 파싱할 수 있어 불필요한 메뉴나 광고를 걸러낼 수 있습니다.

### DirectoryLoader

디렉터리를 순회하며 조건에 맞는 파일을 한꺼번에 읽습니다.

- `glob`: 대상 파일 패턴. `**/*.md`는 하위 폴더까지 모든 마크다운 파일을 의미합니다.
- `loader_cls`: 개별 파일에 사용할 로더 클래스. 지정하지 않으면 `UnstructuredFileLoader`가
  기본값이며, 이 경우 `unstructured` 패키지가 설치되어 있어야 합니다.
- `loader_kwargs`: 개별 로더에 전달할 인자. 한글 파일은 `{"encoding": "utf-8"}`을
  명시하는 것이 안전합니다.

### PyPDFLoader

PDF를 페이지 단위로 읽습니다. 메타데이터에 페이지 번호가 담겨 출처 추적이 쉽습니다.

## 자주 겪는 문제

로드 결과가 0건이면 대부분 경로 문제입니다. 상대 경로는 노트북 파일의 위치가 아니라
커널의 현재 작업 디렉터리를 기준으로 해석되므로, `os.getcwd()`로 먼저 확인하거나
절대 경로를 쓰는 편이 안전합니다.
