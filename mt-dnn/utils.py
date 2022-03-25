import pandas as pd
import json
# import xml.etree.ElementTree as et
# from datasets import load_dataset
# import re
#
#
# def readQuora() -> pd.DataFrame:
#     quora = pd.read_csv("./data/quora dataset/train.csv")
#     return quora
#
#
# def readSNLI(split:str="train") -> pd.DataFrame:
#     ds = {"premise":[],"hypothesis":[],"label":[]}
#     with open(f"./data/snli_1.0/snli_1.0_{split}.jsonl") as dataJson:
#         data = [json.loads(dataJsonLine) for dataJsonLine in dataJson.readlines()]
#     ds = [{k:v for (k,v) in i.items() if k in ["sentence1","sentence2","gold_label"]} for i in data]
#     return pd.DataFrame(ds)
#
#
# def readRTE(setIndex:int)-> pd.DataFrame:
#     xroot = et.parse(f"./data/rte/rte{setIndex}_dev.xml").getroot()
#     ds = {"premise":[],"hypothesis":[],"label":[]}
#     for i in xroot:
#         premise = i[0].text
#         hypothesis = i[1].text
#         label = i.attrib.get("entailment")
#         label = label if label is not None else i.attrib.get("value")
#         if label == "YES" or label=="TRUE":
#             label = 1
#         elif label=="UNKNOWN":
#             label = 2
#         else:
#             label = 0
#         ds["premise"].append(premise)
#         ds["hypothesis"].append(hypothesis)
#         ds["label"].append(label)
#     # xroot = et.parse(f"./data/rte/rte{setIndex}_test.xml").getroot()
#     # for i in xroot:
#     #     premise = i[0].text
#     #     hypothesis = i[1].text
#     #     label = i.attrib.get("entailment")
#     #     label = label if label is not None else i.attrib.get("value")
#     #     if label == "YES" or label =="TRUE":
#     #         label = 1
#     #     elif label=="UNKNOWN":
#     #         label = 2
#     #     else:
#     #         label = 0
#     #     ds["premise"].append(premise)
#     #     ds["hypothesis"].append(hypothesis)
#     #     ds["label"].append(label)
#     return pd.DataFrame(ds)
