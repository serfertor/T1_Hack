from scibox_api import SciBoxClient

client = SciBoxClient(api_key="sk-QFTyxx7PZeJjc5cBhWygoQ")

models = client.list_models()
print([m.id for m in models.data])

resp = client.chat(
    model="Qwen2.5-72B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "Ты дружелюбный помощник"},
        {"role": "user", "content": "Расскажи анекдот"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

print(resp.choices[0].message.content)
