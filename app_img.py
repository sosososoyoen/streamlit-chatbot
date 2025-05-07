import base64
import streamlit as st
from PIL import Image
from io import BytesIO

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

st.title('패션 추천 봇')
model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

if "image_data_list" not in st.session_state:
    st.session_state.image_data_list = []
    
if images := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files = True):
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

if "messages" not in st.session_state:
    st.session_state.messages = []


    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("어떤 패션 스타일을 추천받고 싶으신가요?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    

    messages = [
    {"type":"text",
        "text": "사람의 전신이 찍혀있는 사진이 한 장 주어집니다. 이 때, 이 사람의 관상과 어울리는 패션 스타일을 추천해주세요."},
    ]
    

    for m in st.session_state.messages:
        print(m)
        # if m["role"] == "user":
        #     messages.append({"role": "user", "content": m["content"]})
        # else:
        #     messages.append({"role": "assistant", "content": m["content"]})


    for image_data in st.session_state.image_data_list:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_data['format']};base64,{image_data['base64']}"}
        })



    with st.chat_message("assistant"):
        message = HumanMessage(content=messages)
        result = model.invoke([message])
        response = result.content
        st.markdown(response)
