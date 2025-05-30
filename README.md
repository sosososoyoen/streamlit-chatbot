## 프로젝트 소개

[streamlit-app_pdf-2025-05-23-13-05-29.webm](https://github.com/user-attachments/assets/171f0876-3888-4c3d-a925-ca8087db2fd6)

DEMO : https://app-chatbot-gsxjb9ahrdrkfwrwypwruy.streamlit.app/

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![langchain](https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

PDF와 open ai 키를 입력하면 업로드한 파일을 기반으로 질의응답을 해주는 챗봇

기술 스택 : Python, LangChain, Streamlit

## 프로젝트의 목표

1.  다양한 RAG 기법, Langchain 익히기

2. LLM 애플리케이션 평가 파이프라인 구축하기
3. 
4. GPT-4o-mini나 GPT-4.1-nano 모델에서 할루시네이션을 줄이고, 퀄리티 높은 답변을 생성하는 것


## 기술적 도전 

### 대화 기록을 context로 처리하기
1. `ConversationalRetrievalChain` 을 사용해서 대화 기록을 참조
2. 컨텍스트로 쓰인 원본 문서를 return 해서 출처 표기 기능을 만들 수 있다.

### 다양한 RAG 기법 사용하기
* **멀티 쿼리 검색기**
  * 사용자의 질문을 다양한 유사 질문으로 재생성하여 좀 더 문맥을 잘 이해한 답변을 제공한다.
* **앙상블 검색기**
  * 여러 검색기를 합쳐서 쓰는 검색기
  * 단어 출현 빈도를 체크하는 Sparse 검색기 + 맥락을 고려하는 Dense 검색기의 결합한 후 재정렬
 
### LangSmith로 LLM 애플리케이션 추적 평가 파이프라인 만들기
* openeval 라이브러리로 LLM-as-Judge

 
## ✨ LLM 애플리케이션 평가 지표

| Experiment                 |   Bleu   | Correctness | Helpfulness |  Meteor  |  Rouge   | tokens     | P50 Latency(s) |
| -------------------------- | :------: | :---------: | :---------: | :------: | :------: | ---------- | -------------- |
| gpt-4.1-nano + multi-query |   0.15   |    0.40     |    0.60     |   0.37   |   0.19   | **55,628** | 9.497          |
| gpt-4o-mini + multi-query  | **0.18** |  **0.60**   |  **0.95**   |   0.45   | **0.24** | 47,226     | **12.083**     |
| gpt-4.1-nano + ensemble    |   0.14   |    0.45     |    0.47     |   0.36   |   0.17   | 39,132     | 8.626          |
| gpt-4o-mini + ensemble     |   0.15   |    0.60     |    0.85     |   0.42   |   0.21   | 37,905     | 9.956          |
| gpt-4.1-nano + dense       |   0.15   |    0.45     |    0.85     |   0.37   |   0.19   | 30,864     | 7.939          |
| gpt-4o-mini + dense        | **0.18** |    0.50     |    0.90     | **0.46** | **0.24** | 30,428     | 9.297          |
| gpt-4o                     |   0.12   |    0.20     |    0.80     |   0.31   |   0.16   | 7,239      | 6.076          |
| gpt-4.1-nano               |   0.14   |    0.20     |    0.90     |   0.31   |   0.17   | 6,559      | 4.843          |
| gpt-4o-mini                |   0.12   |    0.30     |    0.90     |   0.33   |   0.16   | 7,110      | 7.896          |

### 평가 기준 설명

**휴리스틱 평가 지표**
* BLEU : 생성된 텍스트가 참조 텍스트와 얼마나 유사한지 측정
* Rouge : 생성된 텍스트가 참조 텍스트의 중요 키워드를 얼마나 포함하는지 측정
* METEOR : 단순 단어 매칭 외에도 동의어 매칭, 패러프레이징 등 다양한 언어학적 요소를 고려

**LLM-as-Judge 지표**
* Correctness : 생성된 답변이 실제 답변과 얼마나 유사하거나 정확한가
* Helpfulness : 생성된 응답이 사용자 입력을 얼마나 잘 처리하는가

### 분석

**1위: gpt-4o-mini + multi-query retriever**
- 모든 지표에서 고른 성장을 보여 평균 0.48로 압도적 1위.
- 특히 Helpfulness(0.95)와 Correctness(0.60)가 가장 높아 ‘정확하면서도 친절한’ 답변을 잘 뽑아냄.


**2~3위: dense vs ensemble (gpt-4o-mini 계열)**
- **Dense retriever** 조합(0.456)은 Bleu·Meteor·Rouge 모두 최상위권, Helpfulness(0.90)·Correctness(0.50)도 견고.
- **Ensemble retriever** 조합(0.446)은 Correctness(0.60)가 최고지만 Helpfulness(0.85)가 약간 낮아 2위와 근소한 차이.

**4위: gpt-4.1-nano + dense retriever**
- gpt-4o-mini 계열보단 평균이 낮지만, gpt-4.1-nano 중에서는 제일 균형 잡힌 조합(0.402).

**중하위권 (5~7위): 기본 모델 vs gpt-4.1-nano + multi-query**
- **기본 gpt-4o-mini**(0.362)와 **gpt-4.1-nano-59c63224**(0.344)는 retrieval 전에도 Helpfulness가 0.90으로 높음.
- **gpt-4.1-nano + multi-query**(0.342)는 정확도 개선(0.40)에도 불구하고 Helpfulness 하락(0.60) 탓에 순위가 밀림.

**꼴찌권: gpt-4o & gpt-4.1-nano + ensemble**
- Avg 0.318로 타 모델 대비 낮은 편.
- 특히 ensemble 조합은 과도한 앙상블로 Helpfulness(0.47)가 크게 떨어져 전반적 성능이 저하됨.

**결론:**
- **최고 성능**은 **gpt-4o-mini + multi-query retriever**
- **토큰 수 vs 속도 vs 품질 균형**을 중시할 땐 **gpt-4o-mini + dense retriever**가 무난한 선택
- gpt-4.1-nano 계열은 dense 조합만 고려해 볼 만함.
