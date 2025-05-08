import base64
import uuid
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages

st.title('👗 패션 추천 봇')

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "image_data_list" not in st.session_state:
    st.session_state.image_data_list = []
config = {"configurable": {"session_id": st.session_state.session_id}}


def get_chat_history(sid: str) -> InMemoryChatMessageHistory:
    chats = st.session_state.chats
    if sid not in chats:
        chats[sid] = InMemoryChatMessageHistory()
    return chats[sid]


session_id = st.session_state.session_id

builder = StateGraph(state_schema=MessagesState)
model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

SYSTEM_PROMPT = SystemMessage(
    content=(
        "당신은 패션 스타일 전문 AI 어시스턴트입니다. "
        "올라온 전신 사진을 보고 그 사람에게 어울리는 스타일을 추천해주세요."
    )
)


def call_model(state: MessagesState, config: RunnableConfig) -> list[BaseMessage]:
    chat_history = get_chat_history(config["configurable"]["session_id"])
    messages = list(chat_history.messages) + state["messages"]
    ai_message = model.invoke(messages)
    chat_history.add_messages(state["messages"] + [ai_message])
    return {"messages": ai_message}


builder.add_edge(START, "model")
builder.add_node("model", call_model)
graph = builder.compile()

if images := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg', 'webp'],
                              accept_multiple_files=True):
    # 파일 확장자와 실제 이미지 형식 검증
    for image in images:
        img = Image.open(image)
        if img.format.lower() not in ['png', 'jpeg', 'jpg', 'webp']:
            st.error("지원되지 않는 이미지 형식입니다. 지원되는 형식: png, jpg, jpeg")
        else:
            st.image(img)

            # 이미지를 메모리 버퍼에 저장 후 Base64 인코딩
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            st.session_state.image_data_list.append({"format": img.format.lower(), "base64": image_base64})

# 챗 내역 렌더링
if "chats" in st.session_state:
    chat_history = get_chat_history(st.session_state.session_id)
    for msg in chat_history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)


def handle_submit():
    user_input = st.session_state.user_input
    if not user_input:
        return

    chat_history = get_chat_history(st.session_state.session_id)

    messages = [{"role": "user", "type": "text", "content": user_input}]

    for image_data in st.session_state.image_data_list:
        messages.append({
            "role": "user",
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_data['format']};base64,{image_data['base64']}"}
        })

    res = graph.invoke({"messages": messages}, config)
    res["messages"][-1].pretty_print()


st.chat_input(
    placeholder="질문을 입력하고 Enter를 눌러주세요…",
    key="user_input",
    on_submit=handle_submit,
)
