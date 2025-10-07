import os
from openai import OpenAI


API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=API_KEY,
)



def call_openai(prompt: str, system_prompt: str = None, model: str = "gpt-4") -> str:
    """调用OpenAI API获取结果"""


    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1000  # 设置最大token数
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    print(call_openai('你好'))
