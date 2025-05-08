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
if "enable_search" not in st.session_state:
    st.session_state.enable_search = False
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/sosososoyoen/streamlit-chatbot)"
    st.session_state.enable_search = st.checkbox("🔍검색", value=st.session_state.enable_search)

st.title("👗Langchain + OpenAI + Image")
st.caption("🚀 업로드한 이미지를 기반으로 답변이 생성됩니다.")



# 서치 rag
search_tool = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"])

prompt_template = PromptTemplate.from_template("""
Answer the question based on the context below. Cite sources if relevant.

이전 대화 내용(History):
{history}

질문: {user_input}

답변:""")


def generate_caption():
    if not st.session_state.image_data_list:
        return ""
    img_msgs = [{"role": "user", "content": [{"type": "text", "text": {"content": "위 이미지들을 설명하는 캡션을 한글로 만들어주세요."}}]}]
    for image_data in st.session_state.image_data_list:
        img_msgs[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_data['format']};base64,{image_data['base64']}"}
        })
    chain = (model | StrOutputParser())
    return chain.invoke(img_msgs)


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
    st.session_state.messages = [
        SystemMessage(content="당신은 패션 스타일 전문 AI 어시스턴트입니다. 올라온 사진을 보고 그 사람에게 어울리는 스타일을 추천해주세요.")
    ]

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# — 채팅 입력 & 토글을 같은 행에 —
prompt = st.chat_input("어떤 패션 스타일을 추천받고 싶으신가요?")


if prompt:
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()
    model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])
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

    if st.session_state.enable_search:
        history = serialize_history(st.session_state.messages, last_n=5)
        final_prompt = prompt_template.format(
            history="\n".join(history),
            user_input=prompt,
        )
        search_results = search_docs(final_prompt, 3)
        print(search_results)

    for image_data in st.session_state.image_data_list:
        img_msgs[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_data['format']};base64,{image_data['base64']}"}
        })

    with st.chat_message("assistant"):
        result = model.invoke(msgs + img_msgs)
        st.session_state.messages.append({"role": "assistant", "content": result.content})
        response = result.content
        st.markdown(response)
