from scibox_api import SciBoxClient

client = SciBoxClient(api_key="sk-QFTyxx7PZeJjc5cBhWygoQ")

models = client.list_models()
print([m.id for m in models.data])

resp = client.embeddings(
    inputs=[
        "Здравствуйте! аш университет и музей? Хотим включить в наш отчет о поездке Если по этому вопросу писать на другой контакт, просьба поделиться"
    ]
)

print(len(resp.data), len(resp.data[0].embedding))