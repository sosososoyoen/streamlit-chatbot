import logging

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import chromadb
import streamlit as st
import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import CacheBackedEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from loguru import logger
import os

os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

store = InMemoryByteStore()
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


def main():
    st.title("🐬 PDF QnA Bot")
    model = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-4.1-nano"))

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "progress" not in st.session_state:
        st.session_state.progress = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your PDF, docx here and click", type=["pdf", "docx"],
                                          accept_multiple_files=True)
        open_ai_key = st.text_input("OpenAI API Key", type="password")
        process = st.button("Process")

    if process:
        if not open_ai_key:
            st.info("Please enter your OpenAI API key.")
            st.stop()
        if not uploaded_files:
            st.info("파일을 업로드 해주세요.")
            st.stop()
        with st.spinner("🫧 파일 읽는 중..."):
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            cached_embedder = get_cached_embeddings(open_ai_key)
            vectorestore = get_vectorstore(text_chunks, cached_embedder)
            sparse_retriever = get_sparse_retriever(text_chunks)
            ensemble_retriever = get_ensemble_retriever(sparse_retriever, vectorestore.as_retriever())

        st.session_state.conversation = get_conversation_chain(ensemble_retriever, open_ai_key, model)

        st.session_state.processComplete = True

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message.get("source_documents"):
                    with st.expander("참고 문서 확인"):
                        for doc in message["source_documents"]:
                            st.markdown(f"{doc.metadata['source']} 📄p.{doc.metadata['page_label']}",
                                        help=doc.page_content)

    history = StreamlitChatMessageHistory(key="chat_messages")

    if st.session_state.processComplete is None:
        st.info("파일과 open API Key를 입력하고 Process 버튼을 눌러주세요.")
        st.stop()

    # chat
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("🐬 파일에서 대답 찾는 중..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(f"{doc.metadata['source']} p.{doc.metadata['page_label']}", help=doc.page_content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "source_documents": source_documents})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_cached_embeddings(open_ai_key):
    embeddings = OpenAIEmbeddings(api_key=open_ai_key)
    return embeddings


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded file: {file_name}")
        if '.pdf' in file_name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in file_name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in file_name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=tiktoken_len)
    text_chunks = text_splitter.split_documents(text)
    return text_chunks


def get_vectorstore(text_chunks, embedder):
    client = chromadb.Client()
    client.clear_system_cache()
    persist_directory = "./chroma_db"
    vectorestore = Chroma.from_documents(text_chunks, embedder, persist_directory=persist_directory,
                                         collection_name="pdf_docs")
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


@traceable(run_type="llm")
def get_conversation_chain(retriever, open_ai_key, model):
    llm = ChatOpenAI(model=model, api_key=open_ai_key, temperature=0)
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


@traceable(run_type="retriever")
def get_multiquery_retriever(vectorestore, model):
    llm = ChatOpenAI(model=model, api_key=st.secrets["OPENAI_KEY"], temperature=0)
    retriever = vectorestore.as_retriever(search_type="mmr")
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )
    return multiquery_retriever


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    main()

summary_prompt = """\
Below is an excerpt from a paper retrieved and extracted using RAG:
{context}

Based on the above content, please summarize the following items in Korean:
1. Purpose of the paper
2. Major contributions
3. Methodology used
4. Key results
5. Limitations and future research directions

For each item, you **must** include:
- Relevant metrics and their exact values
- Section and page references (e.g., §4.1, p.5)
- Specific numerical results (e.g., EM = 44.5 %, Bleu-1 = 40.8)

(Use a subheading for each item and answer in 3–5 sentences.)

"""
