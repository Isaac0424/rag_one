# 텍스트 분할(Text Splitter)

로드한 문서는 대개 그대로 쓰기엔 너무 깁니다. 임베딩 모델에는 입력 길이 제한이 있고,
너무 긴 청크는 검색 정확도를 떨어뜨리기 때문에 적절한 크기로 잘라야 합니다.

## 핵심 파라미터

- `chunk_size`: 청크 하나의 최대 길이
- `chunk_overlap`: 인접한 청크가 겹치는 길이. 문장이 경계에서 잘려 문맥이 끊기는 것을
  막아줍니다. 보통 `chunk_size`의 10~20% 정도를 씁니다.

## RecursiveCharacterTextSplitter

가장 널리 쓰이는 기본 선택지입니다. 구분자 목록을 순서대로 시도하며 분할합니다.

기본 구분자는 문단(`\n\n`) → 줄바꿈(`\n`) → 공백(` `) → 글자 순입니다.
문단 단위로 먼저 자르고, 그래도 `chunk_size`를 넘으면 더 작은 단위로 내려가는 방식이라
의미 단위를 최대한 보존합니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
```

## 그 외 분할기

**CharacterTextSplitter** — 단일 구분자로만 자릅니다. 단순하지만 구분자가 없는
텍스트에서는 `chunk_size`를 훌쩍 넘는 청크가 나올 수 있습니다.

**MarkdownHeaderTextSplitter** — 마크다운 헤더를 기준으로 나누고, 헤더 정보를
메타데이터에 담아줍니다. 문서 구조가 잘 잡힌 자료에 효과적입니다.

**RecursiveCharacterTextSplitter.from_language()** — 프로그래밍 언어의 문법 구조
(함수, 클래스 경계)를 고려해 소스 코드를 자릅니다.

## 크기 선택 기준

정답은 없고 문서 성격에 따라 달라집니다. 청크가 작으면 검색은 정밀해지지만 문맥이
부족해지고, 크면 문맥은 풍부하지만 관련 없는 내용이 섞여 검색 정밀도가 떨어집니다.
실제 질문 몇 개로 검색 결과를 직접 확인하며 조정하는 것이 가장 확실합니다.
