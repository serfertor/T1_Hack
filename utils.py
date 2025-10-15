import pandas as pd
import time

import scibox_api
import nlp_entry


def convert_embeddings(path):
    df = pd.read_excel(path)
    new_col = []
    pre = nlp_entry.QuestionPreprocessor(remove_stopwords=True)
    client = scibox_api.SciBoxClient(api_key="sk-QFTyxx7PZeJjc5cBhWygoQ")
    for i in df["Пример вопроса"]:
        processed = pre.preprocess(str(i))["normalized"]
        resp = client.embeddings(
            inputs=processed
        )
        new_col.append(resp)
        #time.sleep(5)

    df["embedding_bge_m3"] = new_col
    df.to_csv("out_csv", index=False)


if __name__ == "__main__":
    convert_embeddings("E:/T1_Hack/data.xlsx")
