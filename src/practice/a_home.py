import streamlit as st
from utils.sidebar import sidebar


sidebar()

st.title("🎠 Echo Home")

if "messages" not in st.session_state:
    st.session_state.home_messages = [
        {"role": "assistant", "content": "이 곳은 streamlit 연습용 챗봇입니다. api 키를 입력해주세요."}
    ]


for message in st.session_state.home_messages:
    # with : 해당 블록 안에서 실행
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# := 값을 변수에 할당하면서 동시에 그 값을 반환할 수 있도록 해줌.
if prompt := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.home_messages.append({"role": "user", "content": prompt})

    response = f"{prompt}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.home_messages.append({"role": "assistant", "content": response})


