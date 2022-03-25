from datasets import load_dataset
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader,Dataset,ConcatDataset



"""
task=0 -> single sentence classification
tast=1 -> (pairwise sentence classification) NLI  entailment - neutral - contradiction
tast=2 -> QNLI  Relevance ranking  (question & candidate answer)

 """


class DatasetsLoader:
    def __init__(self,tokenizer:PreTrainedTokenizer):
        self.tok = tokenizer

    def MapTokens2SentencesSNLI(self,i:dict,task:int=1)->dict:
        d1 = {k:v for (k,v) in self.tok(i["premise"],text_pair=i["hypothesis"],truncation=True,max_length=512).items()}
        d1["task"] = [task]*len(i["sentence1"])
        return d1

    def MapTokens2SentencesQNLI(self,i:dict,task:int=2)->dict:
        d1 = {k:v for (k,v) in self.tok(i["question"],text_pair=i["answer"],truncation=True,max_length=512).items()}
        d1["task"] = [task]*len(i["question"])
        return d1

    def MapTokens1Sentence(self,i:dict)->dict:
        d = self.tok(i["sentence"],truncation=True,max_length=512)
        d["task"] = [0]* len(i["sentence"])
        return d

    def loadDS(self,task)->(Dataset,Dataset,Dataset):
        ds = None
        if task == 0:
            ds = load_dataset('glue', 'sst2')
            ds = ds.map(lambda x: self.MapTokens1Sentence(x,task),batched=True)
        elif task == 1:
            ds = load_dataset('glue','mnli')
            ds = ds.map(lambda x: self.MapTokens2SentencesSNLI(x,task),batched=True)
        elif task == 2:
            ds = load_dataset('glue', 'qnli')
            ds = ds.map(lambda x: self.MapTokens2SentencesQNLI(x,task),batched=True)
        else:
            assert task in [0,1,2]
            return
        ds.set_format("torch",columns=['attention_mask','input_ids','label','task','token_type_ids'])
        return ds["train"],ds["validation"],ds["test"]

    def getConcatedDataset(self,batch_size:int=32)->(DataLoader,DataLoader,DataLoader):
        sst2Train,sst2Validation,sst2Test = self.loadDS(0)
        snliTrain,snliValidation,snliTest = self.loadDS(1)
        qnliTrain,qnliValidation,qnliTest = self.loadDS(2)

        MTTrain = ConcatDataset([sst2Train,snliTrain,qnliTrain])
        MTValidation = ConcatDataset([sst2Validation,snliValidation,qnliValidation])
        MTTest = ConcatDataset([sst2Test,snliTest,qnliTest])

        return DataLoader(MTTrain,batch_size=32,shuffle=True)\
            , DataLoader(MTValidation,batch_size=32,shuffle=True)\
            , DataLoader(MTTest,batch_size=32,shuffle=True)





