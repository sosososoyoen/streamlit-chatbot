import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
import nltk
from huggingface_hub import login as hf_login
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.community.kiwi_tokenizer import KiwiTokenizer
from langsmith import Client

nltk.download('wordnet')
from langsmith.schemas import Run, Example
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from sentence_transformers import SentenceTransformer, util
import os
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT
)

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "testset.json")
with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Define the input and reference output pairs that you'll use to evaluate your app
client = Client()
dataset_name = "Chatbot QA Example Dataset"
model_id = "gpt-4.1-nano"


def get_model():
    if "gpt" in model_id:
        model = ChatOpenAI(model=model_id, api_key=st.secrets["OPENAI_KEY"], temperature=0)
        return model
    else:
        hf_login(st.secrets["HF_KEY"])
        repo_id = "google/gemma-3-1b-it"
        # hf = HuggingFacePipeline.from_model_id(
        #     model_id=repo_id,
        #     task="text-generation",
        #     pipeline_kwargs={"max_new_tokens": 10},
        # )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=256,
            do_sample=False
        )
        hf = HuggingFacePipeline(pipeline=text_gen)
        print(hf)
        return hf


llm = get_model()

examples = [
    {
        "inputs": {"question": item["question"]},
        "outputs": {"answer": item["answer"]},
    }
    for item in data
]

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=examples)

kiwi_tokenizer = KiwiTokenizer()

# 토크나이저 병렬화 설정(HuggingFace 모델 사용)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def rouge_evaluator(metric: str = "rouge1") -> dict:
    # wrapper function 정의
    def _rouge_evaluator(run: Run, example: Example) -> dict:
        # 출력값과 정답 가져오기
        student_answer = run.outputs.get("answer", "")
        reference_answer = example.outputs.get("answer", "")

        # ROUGE 점수 계산
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True, tokenizer=KiwiTokenizer()
        )
        scores = scorer.score(reference_answer, student_answer)

        # ROUGE 점수 반환
        rouge = scores[metric].fmeasure

        return {"key": "ROUGE", "score": rouge}

    return _rouge_evaluator


def bleu_evaluator(run: Run, example: Example) -> dict:
    # 출력값과 정답 가져오기
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # 토큰화
    reference_tokens = kiwi_tokenizer.tokenize(reference_answer, type="sentence")
    student_tokens = kiwi_tokenizer.tokenize(student_answer, type="sentence")

    # BLEU 점수 계산
    bleu_score = sentence_bleu([reference_tokens], student_tokens)

    return {"key": "BLEU", "score": bleu_score}


def meteor_evaluator(run: Run, example: Example) -> dict:
    # 출력값과 정답 가져오기
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # 토큰화
    reference_tokens = kiwi_tokenizer.tokenize(reference_answer, type="list")
    student_tokens = kiwi_tokenizer.tokenize(student_answer, type="list")

    # METEOR 점수 계산
    meteor = meteor_score.meteor_score([reference_tokens], student_tokens)

    return {"key": "METEOR", "score": meteor}


def semscore_evaluator(run: Run, example: Example) -> dict:
    # 출력값과 정답 가져오기
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # SentenceTransformer 모델 로드
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # 문장 임베딩 생성
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    reference_embedding = model.encode(reference_answer, convert_to_tensor=True)

    # 코사인 유사도 계산
    cosine_similarity = util.pytorch_cos_sim(
        student_embedding, reference_embedding
    ).item()

    return {"key": "sem_score", "score": cosine_similarity}


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    return eval_result


def helpfulness_evaluator(inputs: dict, outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=RAG_HELPFULNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="helpfulness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
    )
    return eval_result


# 응답 로직
# template = """
#       You are an assistant for question-answering tasks.
#   Please write your answer in a markdown table format with the main points.
#   Answer in Korean.
#
#   #Example Format:
#   (brief summary of the answer)
#   (table)
#   (detailed answer to the question)
#
#
#   #Question:
#   {question}
#     """
template = """
      You are an assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  Please write your answer in a markdown table format with the main points.
  Answer in Korean.

  #Example Format:
  (brief summary of the answer)
  (table)
  (detailed answer to the question)


  #Question:
  {question}

  #Context:
  {context}
    """


embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"])


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(text)
    return text_chunks


def get_vectorstore(text_chunks, embedder):
    persist_directory = "./chroma_db"
    vectorestore = Chroma.from_documents(text_chunks, embedder, persist_directory=persist_directory,
                                         collection_name="rag-eval")
    return vectorestore


def get_sparse_retriever(text_chunks):
    return BM25Retriever.from_documents(
        text_chunks,
    )


def get_ensemble_retriever(sparse, dense):
    return EnsembleRetriever(
        retrievers=[sparse, dense],
        weights=[0.5, 0.5],
    )


def get_multiquery_retriever(vectorestore):
    retriever = vectorestore.as_retriever(search_type="mmr")
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )
    return multiquery_retriever


def target(inputs: dict) -> dict:
    files_text = get_text()
    text_chunks = get_text_chunks(files_text)
    vectorestore = get_vectorstore(text_chunks, embeddings)
    sparse_retriever = get_sparse_retriever(text_chunks)
    multiquery_retriever = get_multiquery_retriever(vectorestore)
    # ensemble_retriever = get_ensemble_retriever(sparse_retriever, vectorestore.as_retriever())
    chain = get_conversation_chain(multiquery_retriever)
    # chain = PromptTemplate.from_template(template) | llm | StrOutputParser()
    # result = chain.invoke({"question": inputs["question"]})
    # return {"answer": result}
    result = chain({"question": inputs["question"]})
    return {"answer": result["answer"]}


def get_text():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "test-pdf.pdf")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()

    return documents


def get_conversation_chain(retriever):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(template)}
    )
    return conversation_chain


# 평가 ㄱㄱ

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[
        rouge_evaluator(metric="rougeL"),
        bleu_evaluator,
        meteor_evaluator,
        correctness_evaluator,
        helpfulness_evaluator
    ],
    experiment_prefix=f"{model_id} + multi-query retriever",
    max_concurrency=2,
)
