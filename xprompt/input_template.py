import torch
import torch.nn as nn
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from xprompt.soft_prompt import SoftPrompt
from xprompt import logger

class InputTemplate():
    def __init__(self,
                tokenizer:PreTrainedTokenizer,
                placeholder_mapping:  dict = {'soft':"_",'mask_token':'<mask>','<text_a>':'text_a','<text_b>':'text_b'}):
        self.tokenizer=tokenizer
        self.placeholder_mapping=placeholder_mapping
        self.prompt_pool=[]       
        self.InputData=None

    def build_prompt_pool(self,
                  path: str,
                  pretrain_embedding: nn.Embedding
                 ):
        r'''
        Read the template from a local file and then build prompt pool.
        '''
        with open(path, 'r') as fin:
            templates = fin.readlines()
            for t in templates:
                prompt_text=[]
                for s in t.split():
                    if s not in ["<text_a>","<text_b>"]:
                        prompt_text.append(s)
                prompt=SoftPrompt(prompt_text,self.tokenizer,self.placeholder_mapping,pretrain_embedding)
                logger.info(f"parsing prompt: {t}")
                prompt.parse_text()
                self.prompt_pool.append(prompt)
        return self

if __name__=='__main__':
    tokenizer=BertTokenizer.from_pretrained("/home/yanzhongxiang/XPROMPT/huggingface_models/bert-base-uncased")
    plm= BertForMaskedLM.from_pretrained("/home/yanzhongxiang/XPROMPT/huggingface_models/bert-base-uncased")
    embed_weight=plm.get_input_embeddings().weight.detach().numpy()
    P=InputTemplate(tokenizer=tokenizer)
    P.build_prompt_pool("prompt_source/manual_prompt.txt",embed_weight)
    print(len(P.prompt_pool))