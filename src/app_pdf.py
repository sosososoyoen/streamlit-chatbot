import streamlit as st
import tiktoken
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from loguru import logger


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
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorestore = get_vectorestore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorestore, open_ai_key, model)

        st.session_state.processComplete = True

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요~ PDF QnA Bot입니다."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message.get("source_documents"):
                    with st.expander("참고 문서 확인"):
                        for doc in message["source_documents"]:
                            st.markdown(f"{doc.metadata['source']} 📄p.{doc.metadata['page_label']}", help=doc.page_content)

    history = StreamlitChatMessageHistory(key="chat_messages")

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
                st.session_state.messages.append({"role": "assistant", "content": response, "source_documents": source_documents})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


@st.cache_resource
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

def get_vectorestore(text_chunks):
    persist_directory = "./chroma_db"
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"])
    vectorestore = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory)
    return vectorestore


def get_conversation_chain(vectorestore, open_ai_key, model):
    llm = ChatOpenAI(model=model, api_key=st.secrets["OPENAI_KEY"], temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorestore.as_retriever(search_type="mmr"),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain


if __name__ == "__main__":
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
