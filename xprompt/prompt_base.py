import torch.nn as nn
from numpy import lookfor
from typing import *
from abc import abstractmethod
from transformers.tokenization_utils import PreTrainedTokenizer
from xprompt import logger

class PromptBase():
    def __init__(self,
                text: List[str],
                tokenizer: PreTrainedTokenizer,
                placeholder_mapping: dict,
                pretrain_embedding,) -> None:
        self.text=text
        self.tokenizer=tokenizer
        self.placeholder_mapping=placeholder_mapping
        self.raw_embedding=pretrain_embedding
        self.wte=None

    @abstractmethod
    def parse_text(self):
        pass

    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        pass   

