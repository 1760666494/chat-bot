import streamlit as st
from langchain.llms.base import LLM
import dashscope
from dashscope import Generation
import random
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from http import HTTPStatus
import dotenv

# 加载环境变量
dotenv.load_dotenv('../.env')

qwen_apikey = 'sk-369fb75bfd5b46a4b59e753cda200950'
dashscope.api_key = qwen_apikey

def call_with_messages(content):
    messages = [{'role': 'user', 'content': content}]
    response = Generation.call(model="qwen2-72b-instruct",
                                messages=messages,
                                seed=random.randint(1, 10000),
                                result_format='message')
    if response.status_code == HTTPStatus.OK:
        return response["output"]["choices"][0]["message"]["content"]
    else:
        return ('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

class Qwen_API(LLM):
    def __init__(self):
        super().__init__()

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        response = call_with_messages(prompt)
        return response

    @property
    def _llm_type(self) -> str:
        return "Qwen_API"

# 初始化Qwen_API实例
llm = Qwen_API()

# Streamlit 应用程序开始
st.title("Qwen API Chatbot")

# 创建一个文本输入框供用户提问
user_input = st.text_input("You:", "")

# 检查用户是否输入了内容，并在按下Enter键后发送请求
if user_input:
    # 显示用户的输入
    st.text_area("Your message:", value=user_input, height=100, max_chars=None, key='user_message')

    # 调用Qwen_API获取回复
    response = llm._call(user_input)

    # 显示AI的回复
    st.text_area("Qwen's reply:", value=response, height=200, max_chars=None, key='ai_reply')