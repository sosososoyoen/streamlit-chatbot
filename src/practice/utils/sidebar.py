import streamlit as st

def sidebar():
    with st.sidebar:
        if "chatbot_api_key" not in st.session_state:
            st.session_state.chatbot_api_key = ""
        st.page_link("a_home.py", label="Home", icon="🏠")
        st.page_link("pages/app_img.py", label="OpenAI + Image + Tavily search Chatbot", icon="👗")
        st.page_link("pages/app_img_basic.py", label="OpenAI + Image RAG Chatbot", icon="📷")
        st.text_input(
            "OpenAI API Key",
            key="chatbot_api_key",  # 이미 st.session_state에서 관리됨
            type="password"
            )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/sosososoyoen/streamlit-chatbot)"