import openai

# 填你的秘钥
openai.api_key = "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"


# 提问代码
def chat_gpt(prompt):
    # 你的问题
    prompt = prompt

    # 调用 ChatGPT 接口
    model_engine = "text-davinci-003"
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    print(response)


chat_gpt("Python怎么从入门到精通，具体的学习方法是什么？")