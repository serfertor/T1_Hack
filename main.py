import time

import pandas as pd

import nlp_entry
import scibox_api
from scibox_api import SciBoxClient

df = pd.read_excel("E:/T1_Hack/data.xlsx")
new_col = pd.Series()
pre = nlp_entry.QuestionPreprocessor(remove_stopwords=True)
client = scibox_api.SciBoxClient(api_key="sk-QFTyxx7PZeJjc5cBhWygoQ")
for i in df["Пример вопроса"]:
    processed = pre.preprocess(str(i))
    resp = client.embeddings(
        inputs=processed
    )
    new_col = new_col.append(resp)
    print(1)
    time.sleep(60)

df["embedding_bge_m3"] = new_col
df.to_csv("out_csv", index=False)
