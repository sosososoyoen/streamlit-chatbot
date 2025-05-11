import chromadb
import streamlit as st

from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.title("RAG Bot")
model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

new_system = """\
As an expert of strategic management and entrepreneurship, read this paper and answer the following questions precisely and quantitatively, with all the best of yours. But Jargon can be used as is. You do not have to replace them with easy words. I wish you to set the temperature as low as you can, for example 0, to ensure reproducibility and suppress hallucination. (한국어로 정리를 원하시면, Write in Korean 추가해주세요)

1. What is the purpose of the authors in this research?
2. What are the gaps in the prior literature and what are the originalities of this research? Answer in bulletins.
3. What is the research setting and data authors used? What are the main independent and dependent variables in the paper? What analytical model did they use? List in bulletins and explain briefly.
4. What is the contribution of this paper to academia and the industrial sector?
5. What are the limitations of this research? Answer in bulletins.
6. List the important references in this paper with reason.

   * Column names should be ["citation point", "authors", "title", "journal", "year", "doi"]
   * "Citation point" is the reason why the authors thought these are important.
   * "doi"s should be clickable links, by adding "【https://doi.org/】" in front of dois if needed.

Write in Korean

"""

new_user = """\
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

summary_prompt = ChatPromptTemplate(
    input_variables=["context"],
    messages=[
        SystemMessagePromptTemplate.from_template(new_system),
        HumanMessagePromptTemplate.from_template(new_user),
    ],
)

@st.cache_resource
def get_docs(pdf_stream):
    with open('tmp.pdf', 'wb') as f:
        f.write(pdf_stream.getvalue())
    lodaer = PyPDFLoader('tmp.pdf')

    docs = lodaer.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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



if docs := st.file_uploader("Upload your PDF here and click", type="pdf"):
    retriever = get_retriever(docs)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("What is your question about the PDF?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            retirieved_docs = retriever.invoke(prompt)
            user_prompt = summary_prompt.invoke({"context": format_docs(retirieved_docs), "question": prompt})
            result = model.invoke(user_prompt)

            response = result.content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    