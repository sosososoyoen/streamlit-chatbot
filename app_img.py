import base64
import streamlit as st
from PIL import Image, UnidentifiedImageError
from io import BytesIO

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

st.title('패션 추천 봇')
model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])

if image := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg']):
    try:
        # 파일 확장자와 실제 이미지 형식 검증
        img = Image.open(image)
        if img.format.lower() not in ['png', 'jpeg', 'jpg']:
            st.error("지원되지 않는 이미지 형식입니다. 지원되는 형식: png, jpg, jpeg")
        else:
            st.image(img)

            # 이미지를 메모리 버퍼에 저장 후 Base64 인코딩
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            with st.chat_message("assistant"):
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "사람의 전신이 찍혀있는 사진이 한 장 주어집니다. 이 때, 이 사람의 관상과 어울리는 패션 스타일을 추천해주세요."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img.format.lower()};base64,{image_base64}"},
                        },
                    ]
                )
                result = model.invoke([message])
                response = result.content
                st.markdown(response)
    except UnidentifiedImageError:
        st.error("업로드된 파일이 유효한 이미지 형식이 아닙니다. 지원되는 형식: png, jpg, jpeg")