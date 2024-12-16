from langchain.llms.base import LLM
import dashscope
from dashscope import Generation
import random
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from http import HTTPStatus
import dotenv


dotenv.load_dotenv('../.env')

qwen_apikey = 'sk-369fb75bfd5b46a4b59e753cda200950'
dashscope.api_key = qwen_apikey
def call_with_messages(content):
    messages = [{'role': 'user', 'content': content}]
    response = Generation.call(model="qwen2-72b-instruct",
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                seed=random.randint(1, 10000),
                                # 将输出设置为"message"格式
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
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 调用call_with_messages函数，传入prompt参数
        response = call_with_messages(prompt)
        # 返回response
        return response
    # 获取llm_type属性
    @property
    def _llm_type(self) -> str:
        # 返回"Qwen_API"
        return "Qwen_API"
llm = Qwen_API()
print(llm("你好"))