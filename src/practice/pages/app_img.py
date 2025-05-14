import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.sidebar import sidebar


# 설정 및 상수
if "enable_search" not in st.session_state:
    st.session_state.enable_search = False  
sidebar()
openai_api_key = st.session_state.get("chatbot_api_key", "")

query_template = PromptTemplate.from_template("""
당신은 패션 스타일 전문 AI 어시스턴트입니다. 이미지 설명을 보고 그 사람에게 어울리는 스타일을 추천해주세요.
이미지 설명:
{caption}

유저 질문:
{user_input}

위 맥락을 참고해, 다음 형식으로 출력해줘:

답변: <전문적인 스타일 추천 텍스트>
키워드: <검색에 사용할 핵심 키워드 2~3개를 쉼표로 구분하여>
""")



st.title("👗Langchain + OpenAI + Image + Search")
st.caption("🚀 업로드한 이미지를 기반으로 답변이 생성됩니다.")
st.session_state.enable_search = st.checkbox("🔍검색", value=st.session_state.enable_search)
if not openai_api_key:
    st.info("OpenAI API 키를 입력해주세요.")
    st.stop()
model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
query_chain = query_template | model | StrOutputParser()


# 서치 rag
search_tool = TavilySearchResults(max_results=3, api_key=st.secrets["TAVILY_API_KEY"], include_images=True,
)


#함수 정리
def answer_and_search(caption: str, user_input: str, k: int = 3):
    # 2) 답변 + 키워드 생성
    raw = query_chain.invoke({"caption": caption, "user_input": user_input})
    
    # 3) 모델 출력 파싱
    answer = ""
    keywords = []
    for line in raw.split("\n"):
        if line.startswith("답변:"):
            answer = line[len("답변:"):].strip()
        if line.startswith("키워드:"):
            kws = line[len("키워드:"):].strip()
            keywords = [w.strip() for w in kws.split(",") if w.strip()]
    
    query = " ".join(keywords)
    search_results = search_docs(query, k)
    url_content_pairs = [{"url": doc.metadata["source"], "content": doc.page_content} for doc in search_results]

    return answer, keywords, url_content_pairs

def generate_caption():
    if not st.session_state.image_data_list:
        return ""
    img_msgs = [
        {
            "role": "user", "content": [
                "위 이미지들을 설명하는 캡션을 한글로 만들어주세요."
            ]
        }
    ]
    for image_data in st.session_state.image_data_list:
        img_msgs[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_data['format']};base64,{image_data['base64']}"}
        })
    res = model.invoke(img_msgs)

    return res.content


def search_docs(query: str, k: int = 3):
    results = search_tool.invoke(query)
    return [
        Document(page_content=entry["content"], metadata={"source": entry["url"]})
        for entry in results[:k]
    ]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def serialize_history(messages, last_n=5):
    last_chats = messages[-last_n:]
    lines = []
    for m in last_chats:
        if isinstance(m, SystemMessage):
            lines.append(f"**system:** {m.content}")
        else:
            role = m["role"]
            content = m["content"]
            if role == "user":
                lines.append(f"**user:** {content}")
            else:
                lines.append(f"**assistant:** {content}")
    return "\n".join(lines)


if "image_data_list" not in st.session_state:
    st.session_state.image_data_list = []

if images := st.file_uploader("이미지를 업로드해주세요.", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):

    for image in images:
        img = Image.open(image)
        if img.format.lower() not in ['png', 'jpeg', 'jpg']:
            st.error("지원되지 않는 이미지 형식입니다. 지원되는 형식: png, jpg, jpeg")
        else:
            st.image(img)

            # 이미지를 메모리 버퍼에 저장 후 Base64 인코딩
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            st.session_state.image_data_list.append({"format": img.format.lower(), "base64": image_base64})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 키워드 출력
        if "keywords" in message and message["keywords"]:
            st.caption("🔑 검색 키워드: " + ", ".join(message["keywords"]))
        # URL 출력
        if "urls" in message and message["urls"]:
            for pair in message["urls"]:
                st.markdown(f"- [🔗 {pair['content']}]({pair['url']})")



prompt = st.chat_input("어떤 패션 스타일을 추천받고 싶으신가요?")


if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    msgs = []
    img_msgs = [{"role": "user", "content": []}]

    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            msgs.append(msg)
        elif msg["role"] == "user":
            msgs.append(HumanMessage(
                content=msg["content"],
            ))
        else:
            msgs.append(AIMessage(
                content=msg["content"],
            ))

    with st.chat_message("assistant"):
        with st.spinner("이미지 분석 중...."):
            caption    = generate_caption()
        user_input = prompt
        
        if st.session_state.enable_search:
            with st.spinner("검색 중...."):
                answer, keywords, url_content_pairs = answer_and_search(caption, user_input, k=4)
                # 답변 먼저 보여주기
                st.markdown(answer)
                # 추출된 키워드
                st.caption("🔑 검색 키워드: " + ", ".join(keywords))
                # 이미지 출력
                for pair in url_content_pairs:
                    st.markdown(f"- [🔗 {pair['content']}]({pair['url']})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "keywords": keywords,
                    "urls": url_content_pairs
                })
        else:
            # 검색 꺼져있으면 그냥 답변만
            with st.spinner("답변 생성 중...."):
                qa_only = query_chain.invoke({"caption": caption, "user_input": user_input})
                st.markdown(qa_only)
                st.session_state.messages.append({"role": "assistant", "content": qa_only})

