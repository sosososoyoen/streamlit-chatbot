import streamlit as st

st.title("챗봇 연습중임...")

#테스트

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    # with : 해당 블록 안에서 실행
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# := 값을 변수에 할당하면서 동시에 그 값을 반환할 수 있도록 해줌.
if prompt := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
