import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOpenAI(model="gpt-4o-mini",api_key=st.secrets["OPENAI_KEY"]

st.title("gpt bot")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        messages =[]
        for m in st.session_state.messages:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))

        result = model.invoke(messages)
        response = result.content
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

