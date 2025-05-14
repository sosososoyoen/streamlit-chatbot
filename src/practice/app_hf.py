import streamlit as st
from huggingface_hub import login
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


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

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    #     task="text-generation",
    #     device=0, # GPU 사용
    #     model_kwargs={
    #         "trust_remote_code": True, # 모델이 제공하는 코드를 신뢰
    #     },
    #     pipeline_kwargs={
    #         "max_length": 512,
    #         "do_sample": False, # 샘플링 사용x 확률이 높은 토큰 선택
    #     })

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-2b-it",
        task="text-generation",
        device=0,  # 0번 GPU에 load
        pipeline_kwargs={
            "max_new_tokens": 256,  # 최대 256개의 token 생성
            "do_sample": False  # deterministic하게 답변 결정
        }
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