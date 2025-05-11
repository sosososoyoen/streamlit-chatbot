

## 1. 프로젝트 개요

- **목적**: Streamlit 기반 패션 추천 서비스 프로토타입 구현 및 Open LLM 대체 검증
    
- **기간**: 2주 (14일)
    
- **인원**: 1명
    
- **주요 기술 스택**: Python, Streamlit, OpenAI GPT API, Hugging Face Transformers, Chroma DB, CLIP
    

## 2. 주요 목표

1. **프로토타입 구현**: GPT API 활용 전체 플로우 완성
    
2. **추상화 레이어 설계**: LLM 호출 인터페이스 분리
    
3. **평가 체계 구축**: 정량/정성 지표 및 자동화 스크립트
    
4. **Open LLM 전환**: 2B~7B급 모델 통합 및 성능·비용 비교
    

## 3. 시스템 플로우

[1] **유저 취향 수집**  
└─> Streamlit 첫 화면에서 체크박스/라디오 등으로 “캐주얼/페미닌/포멀/스트리트…” 선택  
└─> 저장소:  
• 프로토타입 ➔ `st.session_state`  
• 장기 히스토리 ➔ 벡터 DB 또는 유저 DB

[2] **이미지 업로드 & 분류**  
└─> 전신 사진 또는 옷 사진 업로드  
└─> Vision 분류:  
• CLIP zero-shot (ViT-B/16)  
• HuggingFace `image-classification` (예: `google/vit-base-patch16-224`)  
└─> 결과 예시:  
`{ "item": "denim jacket", "style_tag": ["casual","street"] }`  
└─> 저장소: session_state/유저 DB + 임베딩 후 벡터 DB upsert

[3] **임베딩 & 벡터 DB**  
└─> 유저 취향 텍스트 + 이미지 분류 결과 텍스트 → 임베딩  
• OpenAI Embeddings 또는 Open LLM Embedding API  
└─> 벡터 DB(Chroma 등)에 저장 → 추천 시 검색에 활용

[4] **추천 생성**  
└─> 유저 프로필 벡터와 상품(또는 스타일) 벡터 유사도 검색  
└─> top-k 결과 → LLM 후처리 → UI에 “이렇게 입어보세요: …”, “키워드: …” 표시

[5] **UX 개선 & 반복**  
└─> 추천 품질·속도·비용 모니터링  
└─> 필요시 Vision 모델 파인튜닝 또는 LLM 사이즈 조정

## 4. 기능 구성

1. **유저 취향 수집**: 캐주얼/페미닌/포멀 등 라디오 버튼 UI
    
2. **이미지 업로드 & 분류**: Vision 분류 모델(CLIP/Hugging Face)
    
3. **임베딩 & 벡터 DB**: Embedding → Chroma DB
    
4. **추천 생성**: 벡터 검색 + LLM 후처리
    
5. **평가 시스템**: 자동화 지표 + 휴먼 평가 폼
    

## 5. 시스템 아키텍처

```
[Streamlit UI]
    ├── st.session_state (취향)
    ├── 이미지 업로드 → Vision 분류
    ├── 프로필 + 분류 결과 → Embedding
    └── 벡터 DB → 유사도 검색 → LLM 인터페이스 → UI 표시
```

## 6. 데이터 흐름

1. 사용자 입력(취향, 이미지)
    
2. 이미지 → CLIP 분류 → 태그
    
3. 취향 텍스트 + 태그 → 임베딩
    
4. 벡터 DB upsert & 검색
    
5. LLM 호출 (추천 문장 + 키워드)
    
6. UI 출력 → 평가 스크립트 수집
    

## 7. 평가 지표

- **추천 텍스트**: BLEU, ROUGE, BERTScore, 키워드 F1
    
- **휴먼 평가**: Likert(1–5점), A/B 테스트, 정성 코멘트
    
- **운영 지표**: CTR, Conversion, Retention
    
- **이미지 분류**: Accuracy, Precision/Recall, F1, Confusion Matrix
    

## 8. 일정 계획

|Phase|기간|주요 업무|
|---|---|---|
|Sprint 1|D1–D7|프로토타입 구현, 취향 UI, 이미지 분류, 벡터 DB|
|Sprint 2|D8–D14|추천 로직 완성, 평가 스크립트, Open LLM 전환|

# 스프린트 체크리스트

### Sprint 1: D1–D7

- **Day 1 (D1)**: 리포지토리 초기화 및 가상환경 구성, 프로젝트 구조 설계
    
- **Day 2 (D2)**: Streamlit UI 컴포넌트 구현 - 취향 수집 라디오/체크박스, `st.session_state` 저장 테스트
    
- **Day 3 (D3)**: GPT API 호출 모듈 작성 및 기본 프롬프트/응답 검증
    
- **Day 4 (D4)**: 이미지 업로드 기능 구현 (파일 업로더 UI 및 미리보기)
    
- **Day 5 (D5)**: CLIP 또는 HuggingFace Vision 분류 프로토타입 작성 및 결과 검증
    
- **Day 6 (D6)**: Embedding API 모듈 작성 및 임베딩 테스트 (OpenAI/Open LLM)
    
- **Day 7 (D7)**: Chroma DB 연결 및 upsert/검색 기능 구현, LLM 추상화 레이어 설계
    

### Sprint 2: D8–D14

- **Day 8 (D8)**: 추천 생성 로직 개발 및 UI 표시 개선
    
- **Day 9 (D9)**: BLEU/ROUGE/BERTScore 자동 평가 스크립트 작성
    
- **Day 10 (D10)**: 키워드 F1 평가 스크립트 작성, 휴먼 평가용 데모 데이터셋 준비
    
- **Day 11 (D11)**: 휴먼 평가 설문 폼 UI 구현 및 내부 테스트
    
- **Day 12 (D12)**: Open LLM(2B~7B) 환경 세팅 및 호출 모듈 구현
    
- **Day 13 (D13)**: GPT API vs Open LLM 성능·비용 비교 테스트 및 결과 수집
    
- **Day 14 (D14)**: 이미지 분류 평가 스크립트 작성, 운영 지표 로깅 대시보드 기초 설정, 최종 보고서 및 배포 준비