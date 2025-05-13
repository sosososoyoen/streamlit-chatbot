import re, base64
import streamlit as st
import streamlit.components.v1 as components
from streamlit_pdf_viewer import pdf_viewer

# ==============================================
# 1) PDF 리트리버·체인 캐시 (한번만 생성)
# ==============================================
@st.experimental_singleton
def get_chain_and_retriever(pdf_bytes):
    # (예시) PyPDFLoader → split → 벡터스토어 → retriever → PromptTemplate → LLMChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter     import RecursiveCharacterTextSplitter
    from langchain.embeddings         import OpenAIEmbeddings
    from langchain.vectorstores       import Chroma
    from langchain import PromptTemplate, LLMChain, OpenAI

    loader = PyPDFLoader(pdf_bytes)
    docs   = loader.load()
    chunks = RecursiveCharacterTextSplitter(1000).split_documents(docs)
    store  = Chroma.from_documents(chunks, OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"]))
    retr   = store.as_retriever(search_kwargs={"k":3})
    prompt = PromptTemplate(
        template="""
문맥(Context) 안의 [p.X] 태그를 참고해서 답변할 때 반드시 출처를 달아줘.

문맥:
{context}

질문:
{question}
""",
        input_variables=["context","question"]
    )
    llm   = OpenAI(model="gpt-4o-mini", temperature=0, api_key=st.secrets["OPENAI_KEY"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain, retr

# ==============================================
# 2) 세션 스테이트 초기화 (한번만)
#    — structure-vision 패턴 참고 :contentReference[oaicite:0]{index=0}
# ==============================================
if 'pdf_ref'     not in st.session_state: st.session_state['pdf_ref'] = None
if 'scroll_page' not in st.session_state: st.session_state['scroll_page'] = 1

# ==============================================
# 3) 파일 업로드 (key 고정)
# ==============================================
uploaded = st.file_uploader("PDF 파일 선택", type="pdf", key="pdf_uploader")
if uploaded is not None:
    # 새 파일이면 세션에 저장
    if st.session_state['pdf_ref'] is None or st.session_state['pdf_ref'].name != uploaded.name:
        st.session_state['pdf_ref'] = uploaded
        st.session_state['scroll_page'] = 1

pdf_ref = st.session_state['pdf_ref']
pdf_bytes = pdf_ref.getvalue() if pdf_ref else None

# ==============================================
# 4) 레이아웃: 좌우 컬럼
# ==============================================
col1, col2 = st.columns([1,1])
with col2:
    st.header("👀 PDF 뷰어")
    if pdf_bytes:
        # pdf-viewer 컴포넌트에 scroll_to_page 전달
        pdf_viewer(input=pdf_bytes, width="100%", height=800,
                   scroll_to_page=st.session_state['scroll_page'])
    else:
        st.info("왼쪽에서 PDF를 업로드해주세요.")

with col1:
    st.header("💬 질문 & 답변")
    question = st.text_input("질문을 입력하세요")
    if st.button("질문하기") and pdf_bytes:
        # 5) 캐시된 체인·리트리버 얻기
        chain, retriever = get_chain_and_retriever(pdf_bytes)

        # 6) 문맥 생성
        docs    = retriever.get_relevant_documents(question)
        context = "\n\n".join(
            f"[p.{d.metadata.get('page','?')}] {d.page_content[:200]}…"
            for d in docs
        )

        # 7) 체인 실행
        response = chain.run({"context": context, "question": question})
        st.markdown(response)

        # 8) 답변에서 실제 쓴 [p.X] 태그만 파싱해서 버튼으로
        pages = sorted({int(p) for p in re.findall(r'\[p\.(\d+)\]', response)})
        if pages:
            st.markdown("**📑 출처 페이지:**")
            btn_cols = st.columns(len(pages))
            for col, pg in zip(btn_cols, pages):
                if col.button(f"p.{pg}", key=f"btn_{pg}"):
                    st.session_state['scroll_page'] = pg
