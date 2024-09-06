from volcenginesdkarkruntime import Ark
import os

client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

stream = client.chat.completions.create(
    model="ep-20240711061503-rckh6",
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    stream=True
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()