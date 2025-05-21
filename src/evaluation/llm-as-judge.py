import json
import os

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
from langsmith import Client
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
dataset_name = "RAG QA Example Dataset"
model = "gpt-4o-mini"

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


def get_multiquery_retriever(vectorestore, model):
    llm = ChatOpenAI(model=model, api_key=st.secrets["OPENAI_KEY"], temperature=0)
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
    multiquery_retriever = get_multiquery_retriever(vectorestore, model)
    ensemble_retriever = get_ensemble_retriever(sparse_retriever, vectorestore.as_retriever())
    chain = get_conversation_chain(ensemble_retriever, model)
    result = chain({"question": inputs["question"]})
    return {"response": result["answer"]}


def get_text():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "test-pdf.pdf")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()

    return documents


def get_conversation_chain(retriever, model):
    # llm = get_model()
    llm = ChatOpenAI(model=model, api_key=st.secrets["OPENAI_KEY"], temperature=0)
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
        correctness_evaluator,
        helpfulness_evaluator
    ],
    experiment_prefix=f"{model} + ensemble retriever",
    max_concurrency=2,
)
