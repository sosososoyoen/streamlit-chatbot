import json
import random

import outlines.models as models
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from outlines import generate
from pydantic import BaseModel, ConfigDict


# —————————————————————————————
# 1) 벡터스토어에서 문서 조각 가져오기
# —————————————————————————————
def get_vectorstore():
    embedder = OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"])
    persist_directory = "./chroma_db"
    vectorstore = Chroma(
        collection_name="pdf_docs",
        embedding_function=embedder,
        persist_directory=persist_directory,
    )
    return vectorstore


vectordb = get_vectorstore()
docs = vectordb._collection.get()["documents"]

# —————————————————————————————
# 2) Outlines OpenAI 모델 초기화
# —————————————————————————————
# API 키는 환경변수 OPENAI_API_KEY 로부터 가져옵니다.
model = models.openai(
    "gpt-4o",
    api_key=st.secrets["OPENAI_KEY"],
)


# —————————————————————————————
# 3) 출력 스키마 정의 (Pydantic)
# —————————————————————————————
class QAPair(BaseModel):
    # extra='forbid' 꼭 필요 (OpenAI 구조화 출력 요건)
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str


# —————————————————————————————
# 4) 구조화 생성기(generator) 준비
# —————————————————————————————
qa_generator = generate.json(model, QAPair)

# —————————————————————————————
# 5) 랜덤 샘플링 & QA 생성
# —————————————————————————————
num_samples = 20
samples = random.sample(docs, k=min(num_samples, len(docs)))

qa_pairs = []
for idx, chunk_text in enumerate(samples, start=1):
    prompt = (
        f"다음 문서 내용을 보고, **정확한 질문 하나**와 **그 질문에 대한 답변**을 "
        f"순수 JSON(QAPair)으로만 출력하세요.\n\n"
        f"문서 내용:\n{chunk_text}\n\n"
        f"⚠️ 순수 JSON 외 다른 텍스트를 섞지 말고, 바로 `{{\"question\":...,\"answer\":...}}` 형태로만 응답해주세요."
    )
    qa: QAPair = qa_generator(prompt)
    qa_pairs.append({
        "id": idx,
        "question": qa.question,
        "answer": qa.answer
    })

# —————————————————————————————
# 6) 파일로 저장
# —————————————————————————————
with open("testset.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(qa_pairs)} QA pairs into testset.json")
