import torch
import torch.nn as nn
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import *
from xprompt.prompt_base import PromptBase

class SoftPrompt(PromptBase):
    def __init__(self, text: List[str], tokenizer: PreTrainedTokenizer, placeholder_mapping: dict, pretrain_embedding) -> None:
        super().__init__(text, tokenizer, placeholder_mapping, pretrain_embedding)
        
    def check_prompt(self):
        mask_count=self.text.count(self.placeholder_mapping['mask_token'])
        soft_count=self.text.count(self.placeholder_mapping['soft'])
        assert mask_count>0,"there must exits one mask token at least!"
        return mask_count+soft_count==len(self.text)

    def parse_text(self):
        assert self.check_prompt() is True
        self.wte=self.initialize_embedding()
        print(self.wte)
    
    def initialize_embedding(self, random_range: float = 0.5, initialize_from_vocab: bool = True):
        soft_count=self.text.count(self.placeholder_mapping['soft'])
        if initialize_from_vocab:
            return self.raw_embedding[:soft_count]
        return torch.FloatTensor(soft_count,self.raw_embedding.size(1)).uniform_(-random_range,random_range)

