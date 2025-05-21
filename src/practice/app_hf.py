import streamlit as st
from huggingface_hub import login
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint

st.title("Hugging Face Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# LLM load를 한 번만 실행할 수 있도록 캐싱
@st.cache_resource
def get_model():
    # Hugging Face에 로그인
    hf_token = st.secrets["HF_KEY"]
    login(hf_token)

    llm = HuggingFaceEndpoint(
        repo_id="mlx-community/Phi-3-mini-4k-instruct-4bit",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    return ChatHuggingFace(llm=llm)


model = get_model()

if prompt := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        messages = []
        # 챗 내역들을 모두 포함하여 prompt 준비
        for m in st.session_state.messages:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))
        result = model.invoke(messages)
        print(result)
        response = result.content.split('<start_of_turn>')[-1]
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
