from audioop import add
import torch
from torch.utils.data import TensorDataset
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer 
from data_utils.task_processors import DataProcessor,InputExample,InputFeatures,MnliProcessor,load_examples

class SentencePairDataset(object):
    def __init__(self, processor: DataProcessor, tokenizer: PreTrainedTokenizer,max_seqlen) -> None:
        self.processor= processor
        self.tokenizer=tokenizer
        self.max_seqlen=max_seqlen
        self.truncate_seqlen=max_seqlen
        if self.tokenizer.name_or_path.find("roberta") != -1:
            self.truncate_seqlen-=1

            
    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        label_map= {label : i for i, label in enumerate(self.processor.get_labels()) }
        label_id = label_map[example.label]
        tokens_a = self.tokenizer.encode(example.text_a,add_special_tokens=False)
        tokens_b = self.tokenizer.encode(example.text_b,add_special_tokens=False)
        self._truncate_seq_pair(tokens_a,tokens_b,self.truncate_seqlen-3)
        input_ids=self.tokenizer.build_inputs_with_special_tokens(tokens_a,tokens_b)
        token_type_ids=self.tokenizer.create_token_type_ids_from_sequences(tokens_a,tokens_b)
        attention_mask=[1]*len(input_ids)
        # padding up to the sequence length.
        padding_len=self.max_seqlen-len(input_ids)
        input_ids+=[self.tokenizer.pad_token_id]*padding_len
        token_type_ids+=[0]*padding_len
        attention_mask+=[0]*padding_len
        if(len(input_ids)!=self.max_seqlen):
            print("")
        assert len(input_ids)==self.max_seqlen
        assert len(token_type_ids)==self.max_seqlen
        assert len(attention_mask)==self.max_seqlen
        return InputFeatures(input_ids,attention_mask,token_type_ids,label_id,example.meta)


    def convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.get_input_features(example)
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features   

    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()    


class SentenceSingleDataset(object):
    def __init__(self, processor: DataProcessor, tokenizer: PreTrainedTokenizer,max_seqlen) -> None:
        self.processor= processor
        self.tokenizer=tokenizer
        self.max_seqlen=max_seqlen
        self.truncate_seqlen=max_seqlen
        if self.tokenizer.name_or_path.find("roberta") != -1:
            self.truncate_seqlen-=1

            
    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        label_map= {label : i for i, label in enumerate(self.processor.get_labels()) }
        label_id = label_map[example.label]
        tokens_a = self.tokenizer.encode(example.text_a,add_special_tokens=False)
        tokens_b = self.tokenizer.encode(example.text_b,add_special_tokens=False)
        self._truncate_seq_pair(tokens_a,tokens_b,self.truncate_seqlen-3)
        input_ids=self.tokenizer.build_inputs_with_special_tokens(tokens_a,tokens_b)
        token_type_ids=self.tokenizer.create_token_type_ids_from_sequences(tokens_a,tokens_b)
        attention_mask=[1]*len(input_ids)
        # padding up to the sequence length.
        padding_len=self.max_seqlen-len(input_ids)
        input_ids+=[self.tokenizer.pad_token_id]*padding_len
        token_type_ids+=[0]*padding_len
        attention_mask+=[0]*padding_len
        if(len(input_ids)!=self.max_seqlen):
            print("")
        assert len(input_ids)==self.max_seqlen
        assert len(token_type_ids)==self.max_seqlen
        assert len(attention_mask)==self.max_seqlen
        return InputFeatures(input_ids,attention_mask,token_type_ids,label_id,example.meta)


    def convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.get_input_features(example)
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features   

    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()    
if __name__=='__main__':
    dataset_name='mnli'
    data_dir="/extra/yanzhongxiang/XPROMPT/datasets/MNLI"
    examples=load_examples(dataset_name,data_dir,"train")[:20]
    processor= MnliProcessor()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    Sentences = SentencePairDataset(processor,tokenizer,128)
    features = Sentences.convert_examples_to_features(examples)
    all_input_ids=torch.tensor([f.input_ids for f in features],dtype=torch.long)
    all_input_labels=torch.tensor([f.label for f in features],dtype=torch.long)
    train_data=TensorDataset(all_input_ids,all_input_labels)
    for x,y in train_data:
        pass
