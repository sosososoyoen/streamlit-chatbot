import base64
import uuid
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage

# ── 1) 세션별 히스토리 관리 ──────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "image_data_list" not in st.session_state:
    st.session_state.image_data_list = []


def get_chat_history(sid: str) -> InMemoryChatMessageHistory:
    if sid not in st.session_state.chats:
        st.session_state.chats[sid] = InMemoryChatMessageHistory()
    return st.session_state.chats[sid]


session_id = st.session_state.session_id
history = get_chat_history(session_id)

# ── 2) LangGraph 그래프 정의 ──────────────────────────

workflow = StateGraph(state_schema=MessagesState)
model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

SYSTEM_PROMPT = SystemMessage(
    content=(
        "당신은 패션 스타일 전문 AI 어시스턴트입니다. "
        "올라온 전신 사진을 보고 그 사람에게 어울리는 스타일을 추천해주세요."
    )
)



def call_model(state: MessagesState) -> dict:
    user_msg = state["messages"][0]
    print("유저메세지",prompt_template)
    all_msgs = history.messages + state["messages"]
    res = model.invoke(all_msgs)
    history.add_message(state["messages"] + [res])
    return {"messages": res}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ── 3) Streamlit UI ──────────────────────────
st.title('👗 패션 추천 봇')

uploaded_file = st.file_uploader(
    "본인의 전신이 보이는 사진을 올려주세요!",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)
if uploaded_file:
    st.session_state.image_data_list.clear()
    cols = st.columns(len(uploaded_file))
    for col, file in zip(cols, uploaded_file):
        try:
            img = Image.open(file)
        except UnidentifiedImageError:
            st.error(f"{file.name} 은(는) 지원되지 않는 형식입니다.")
            continue
        col.image(img, use_container_width=True)
        buf = BytesIO()
        img.save(buf, format=img.format)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        st.session_state.image_data_list.append({
            "format": img.format.lower(),
            "base64": b64
        })

for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        content = msg
        print(content)
        # if isinstance(content, list):
        #     for item in content:
        #         if item.get("type") == "text":
        #             st.markdown(item["text"])
        #         elif item.get("type") == "image_url":
        #             st.image(item["image_url"]["url"])
        # else:
        #     st.markdown(content)

config = {"configurable": {"thread_id": session_id}}


def handle_submit():
    user_input = st.session_state.user_input
    st.chat_message("user").write(user_input)
    # multimodal = [
    #     {
    #         "type": "text",
    #         "text": (
    #             "다음은 사용자가 올린 전신 사진입니다. "
    #             "이 사람의 체형과 어울리는 패션 스타일을 추천해주세요."
    #         )
    #     }
    # ]
    # for img in st.session_state.image_data_list:
    #     multimodal.append({
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/{img['format']};base64,{img['base64']}"
    #         }
    #     })
    # multimodal.append({"type": "text", "text": user_input})
    user_msg = HumanMessage(user_input)
    output = app.invoke({"messages": [user_msg]}, config)
    contents = [msg.content for msg in output["messages"]]
    st.chat_message("assistant").write(contents[-1])


st.chat_input(
    placeholder="질문을 입력하고 Enter를 눌러주세요…",
    key="user_input",
    on_submit=handle_submit,
)
