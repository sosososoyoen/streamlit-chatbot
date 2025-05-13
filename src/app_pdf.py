import chromadb
import streamlit as st

from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_pdf_viewer import pdf_viewer
import re


st.title("RAG Bot")

# 1. Set up

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_ref" not in st.session_state:
    st.session_state["pdf_ref"] = None
if "pdf_bytes" not in st.session_state:
    st.session_state["pdf_bytes"] = None
# 페이지 이동을 위한 session state 초기화
if 'page_selection' not in st.session_state:
    st.session_state['page_selection'] = []

chromadb.api.client.SharedSystemClient.clear_system_cache()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

prompt = PromptTemplate.from_template("""
당신은 PDF 문서를 바탕으로 정확하게 답하는 비서입니다.
문맥에는 [p. X] 형태의 페이지 태그가 포함되어 있습니다.
답변할 때 참고한 모든 페이지 태그를 [p. X]로 문장 끝에 삽입해 주세요.


문맥:
{context}

질문:
{question}
"""
)

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



@st.cache_resource
def get_docs(pdf_stream):
    with open('tmp.pdf', 'wb') as f:
        f.write(pdf_stream.getvalue())
    lodaer = PyMuPDFLoader('tmp.pdf')
    docs = lodaer.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    return splits

# @st.cache_resource
def get_vectore_store(_docs):
    splits = get_docs(_docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"]))
    return vectorstore

# @st.cache_resource
# def get_chain(pdf_stream):
#     docs = get_docs(pdf_stream)
#     vectorstore = get_vectore_store(docs)
#     retriever = vectorstore.as_retriever()
#     llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def change_page(page):
    st.session_state['page_selection'] = [page]


uploaded = st.file_uploader("Upload your PDF here and click", type="pdf", key="pdf_uploader")
    

if uploaded is not None:
    if st.session_state.get("pdf_ref") != uploaded:
        print("새로운 PDF 업로드")
        st.session_state["pdf_ref"] = uploaded
        st.session_state["pdf_bytes"] = uploaded.getvalue()
        st.session_state["page_selection"] = [1]  # 초기 페이지 설정

        # pdf_viewer(input=st.session_state["pdf_bytes"], width="100%", height=800, scroll_to_page=1)
    
    with st.spinner("✨ PDF 파일 읽는 중..."):
        retriever = get_vectore_store(st.session_state["pdf_ref"]).as_retriever()
        st.session_state["retriever"] = retriever
    # Chain 초기화
    chain = (
        {"context": st.session_state["retriever"], "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.session_state["chain"] = chain
    
        
    # messages rendering
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("What is your question about the PDF?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Chain 실행
        if "chain" in st.session_state:
            chain = st.session_state["chain"]
            with st.chat_message("assistant"):
                with st.spinner("🐬 PDF에서 대답 찾는 중..."):
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # 출처 페이지 표시
                    # pages = sorted({int(p) for p in re.findall(r'\[p\.(\d+)\]', response)})
                    # if pages:
                    #     st.markdown("**출처 페이지:**")
                    #     cols = st.columns(len(pages))
                    #     for col, page in zip(cols, pages):
                    #         if col.button(label=f"p.{page}", key=f"page_btn_{page}"):
                    #             change_page(page)
        else:
            st.error("⚠️ PDF 파일을 먼저 업로드 해주세요.")


