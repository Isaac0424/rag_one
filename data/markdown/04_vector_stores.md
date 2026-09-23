# 벡터 저장소(Vector Store)

벡터 저장소는 임베딩된 문서 벡터를 보관하고, 질의 벡터와 가장 가까운 것들을
빠르게 찾아주는 데이터베이스입니다.

## 유사도 측정

가장 많이 쓰이는 방식은 코사인 유사도(cosine similarity)로, 두 벡터가 이루는 각도를
기준으로 방향이 비슷할수록 1에 가까운 값을 냅니다. 벡터의 길이가 아니라 방향만 보기
때문에 문서 길이 차이에 덜 민감합니다.

## 주요 선택지 비교

| 저장소 | 실행 방식 | 특징 |
|---|---|---|
| FAISS | 로컬 (메모리) | 매우 빠름, 설치 간단. 학습·프로토타입에 적합 |
| Chroma | 로컬 (파일) | 디스크 영속화 지원. 소규모 실서비스 가능 |
| Pinecone | 클라우드 | 관리형 서비스. 확장성 우수, 유료 |
| Qdrant | 로컬/클라우드 | 메타데이터 필터링이 강력 |
| pgvector | PostgreSQL 확장 | 기존 RDB와 통합 운영 가능 |

학습 단계에서는 FAISS나 Chroma로 시작하는 것을 권합니다.

## 검색 방식

**Similarity Search** — 질문 벡터와 가장 가까운 상위 k개를 반환하는 기본 방식입니다.

**MMR (Maximal Marginal Relevance)** — 유사도만 보면 거의 같은 내용의 청크가
중복으로 뽑히는 문제가 있습니다. MMR은 관련성과 다양성을 함께 고려해 서로 다른
관점의 문서를 섞어 반환합니다.

**메타데이터 필터링** — 특정 출처나 날짜 범위로 검색 대상을 좁힙니다.
`filter={"source": "manual.pdf"}` 같은 형태로 지정합니다.

## Retriever로 변환

벡터 저장소는 `as_retriever()`를 통해 체인에 바로 연결할 수 있는 Retriever가 됩니다.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4},
)
```
