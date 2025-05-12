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




st.title("RAG Bot")

# 1. Set up

if "messages" not in st.session_state:
    st.session_state.messages = []
    
llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])
retriever = None
chain = None


prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
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

def get_retriever(docs):
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    splits = get_docs(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"]))
    retriever = vectorstore.as_retriever()

    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# messages rendering
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if docs := st.file_uploader("Upload your PDF here and click", type="pdf"):
    with st.spinner("✨ PDF 파일 읽는 중..."):
        retriever = get_retriever(docs)
        st.session_state["retriever"] = retriever
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        st.session_state["chain"] = chain

if prompt := st.chat_input("What is your question about the PDF?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "retriever" in st.session_state and "chain" in st.session_state:
        chain = st.session_state["chain"]

        with st.chat_message("assistant"):
            with st.spinner("🐬 PDF에서 대답 찾는 중..."):
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("PDF를 업로드하고 질문을 해주세요.")
    