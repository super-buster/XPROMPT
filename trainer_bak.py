from distutils.command.config import config
import os
from re import T
import sys
import time
from tqdm import tqdm, trange
import math
from datasets import load_dataset
from accelerate import Accelerator
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['TOKENIZERS_PARALLELISM']='true'
sys.path.append(".")
import numpy as np
import dill as pickle
import torch
from transformers import (
    HfArgumentParser,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)

from transformers.utils import check_min_version
from transformers import  (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from typing import List
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.16")


import log
logger=log.get_logger("trainer")

from config import TrainConfig,EvalConfig
from data_utils.task_processors import PROCESSORS,TASKS_MAPPING,load_examples
from data_utils.misc import SentencePairDataset

accelerator = Accelerator()

def prepare_dataset(task_name,data_dir,train_args):
    tokenizer= AutoTokenizer.from_pretrained(train_args.model_type,use_fast=True)
    processor=PROCESSORS[task_name]()
    train_sentences = SentencePairDataset(processor,tokenizer,train_args.max_seqlen)
    if train_args.resume is True:
        logger.info("*****Loading train features... *****")
        with open(os.path.join(train_args.output_dir,task_name),"rb") as f:
            train_features = pickle.load(f)
    else:
        train_examples=load_examples(task_name,data_dir,"train",num_examples=10000)
        train_features = train_sentences.convert_examples_to_features(train_examples)
        logger.info("*****Saving train features... *****")
        with open(os.path.join(train_args.output_dir,task_name),"wb") as f:
            pickle.dump(train_features,f)
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)        
    train_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data= TensorDataset(train_input_ids,train_input_mask,train_token_type_ids,train_label_ids)
    #train_sampler=RandomSampler(train_data)
    train_dataloader=DataLoader(train_data,shuffle=True,batch_size=train_args.per_gpu_train_batch_size,num_workers=train_args.dataload_num_workers)
    eval_examples=load_examples(task_name,data_dir,"dev")
    eval_features = train_sentences.convert_examples_to_features(eval_examples)
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
    eval_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data= TensorDataset(eval_input_ids,eval_input_mask,eval_token_type_ids,eval_label_ids)
    #eval_sampler=RandomSampler(eval_data)
    eval_dataloader= DataLoader(eval_data,batch_size=train_args.per_gpu_eval_batch_size,shuffle=False,num_workers=train_args.dataload_num_workers)
    return len(train_features),train_dataloader,eval_dataloader


if __name__=='__main__':
    parser=HfArgumentParser(TrainConfig)
    train_args=parser.parse_args_into_dataclasses()[0]
    logger.info("Parameters: {}".format(train_args))
    use_cuda=1 if train_args.n_gpu>0 else 0
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    set_seed(train_args.seed)
    if os.path.exists(train_args.output_dir) and os.listdir(train_args.output_dir) and not train_args.overwrite_output_dir:
        raise ValueError("Output directory already exists and not empty!")
    os.makedirs(train_args.output_dir,exist_ok=True)
    source_task='mnli'
    target_task='snli'
    if source_task not in PROCESSORS.keys() or target_task not in PROCESSORS.keys():
        raise ValueError("Choose a valid task!")
    source_processor=PROCESSORS[source_task]()
    target_processor=PROCESSORS[target_task]()
    source_data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasets/"+TASKS_MAPPING[source_task])
    target_data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasets/"+TASKS_MAPPING[target_task])
    model_config = AutoConfig.from_pretrained(train_args.model_type,num_labels=len(target_processor.get_labels()),finetuning_task=target_task)
    if train_args.resume is True and os.path.exists(os.path.join(train_args.output_dir,"pytorch_model.bin")):
        model=AutoModelForSequenceClassification.from_pretrained(train_args.output_dir)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(train_args.model_type,config=model_config)
    _,source_train_dataloader,source_eval_dataloader=prepare_dataset(source_task,source_data_dir,train_args)
    num_examples,target_train_dataloader,target_eval_dataloader=prepare_dataset(target_task,target_data_dir,train_args)

    num_train_step=int(math.ceil(num_examples/train_args.per_gpu_train_batch_size)*train_args.num_train_epochs)
    if num_train_step>train_args.max_steps:
        num_train_step=train_args.max_steps
    
    model.cuda()
    # https://github.com/huggingface/transformers/issues/492
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': train_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': train_args.weight_decay, 'lr': train_args.learning_rate/2.6},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': train_args.weight_decay, 'lr': train_args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': train_args.weight_decay, 'lr': train_args.learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': train_args.learning_rate/2.6},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': train_args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': train_args.learning_rate*2.6},
    ]
    
    # optimizer_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': train_args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    # ]
    optimizer=AdamW(optimizer_parameters,lr=train_args.learning_rate,eps=train_args.adam_epsilon)
    model, optimizer,source_train_dataloader,source_eval_dataloader,target_train_dataloader,target_eval_dataloader= accelerator.prepare(
        model, optimizer,source_train_dataloader,source_eval_dataloader,target_train_dataloader,target_eval_dataloader
    )
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=train_args.warmup_steps,num_training_steps=num_train_step)
    global_step=0
    epoch=0
    model.zero_grad()
    best_eval_accuracy=0
    for _ in trange(train_args.num_train_epochs,desc="Epoch"):
        epoch+=1
        model.train()
        source_train_iterator=source_train_dataloader._get_iterator()
        target_train_iterator=target_train_dataloader._get_iterator()
        tr_loss,tr_domain_loss=0,0
        nb_tr_examples,nb_tr_steps=0,0
        start_time=time.time()
        for step in tqdm(range(int(num_train_step/train_args.num_train_epochs)),desc="Iteration"):
            domain_loss = 0
            # source train
            batch=source_train_iterator.__next__()
            source_label=torch.zeros(batch[0].size(0),dtype=torch.long).cuda()
            batch=tuple(t.cuda() for t in batch)
            batch=tuple(t for t in batch)
            input_ids,attention_mask,token_type_ids,labels=batch
            outputs=model(input_ids= input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,
                        labels=labels,domain_labels=source_label,alpha=train_args.alpha)
            loss= outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            domain_loss = domain_loss + outputs["domain_loss"] if isinstance(outputs, dict) else outputs[-1]
            # target train
            batch=target_train_iterator.__next__()
            batch=tuple(t.cuda() for t in batch)
            input_ids,attention_mask,token_type_ids,labels=batch
            target_label=torch.ones(batch[0].size(0),dtype=torch.long).cuda()
            outputs=model(input_ids= input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,
                        labels=labels,domain_labels=target_label,alpha=train_args.alpha)
            loss= loss+ outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            domain_loss =  domain_loss + outputs["domain_loss"] if isinstance(outputs, dict) else outputs[-1]
            loss+=domain_loss
            if train_args.gradient_accumulation_steps>1:
                loss=loss/train_args.gradient_accumulation_steps
            #loss.backward()
            accelerator.backward(loss)
            tr_loss+=loss.item()
            tr_domain_loss+=domain_loss.item()
            nb_tr_examples+=input_ids.size(0)
            nb_tr_steps+=1
            if (step+1)%train_args.gradient_accumulation_steps==0 or step==len(target_train_dataloader)-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),train_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step+=1
        end_time=time.time()
        model.eval()
        eval_loss,eval_accuracy=0,0
        nb_eval_steps, nb_eval_examples = 0, 0
        source_eval_loss,source_eval_accuracy=0,0
        nb_source_eval_steps, nb_source_eval_examples = 0, 0
        with open(os.path.join(train_args.output_dir,"results_ep"+str(epoch)+".txt"),"w") as f:
            for step,batch in enumerate(tqdm(target_eval_dataloader,desc="Evaluate")):
                batch=tuple(t.cuda() for t in batch)
                #batch=tuple(t for t in batch)
                input_ids,attention_mask,token_type_ids,labels=batch
                with torch.no_grad():
                    outputs=model(input_ids= input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,labels=labels)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                logits= logits.detach().cpu().numpy()
                labels=labels.to('cpu').numpy()
                predictions=np.argmax(logits,axis=1)
                for item in predictions:
                    f.write(str(item)+'\n')
                tmp_eval_accuracy=np.sum(predictions==labels)
                eval_loss+=outputs[0].mean().item()
                eval_accuracy+=tmp_eval_accuracy
                nb_eval_examples+=input_ids.size(0)
                nb_eval_steps+=1
            for step,batch in enumerate(tqdm(source_eval_dataloader,desc="Evaluate")):
                batch=tuple(t.cuda() for t in batch)
                #batch=tuple(t for t in batch)
                input_ids,attention_mask,token_type_ids,labels=batch
                with torch.no_grad():
                    outputs=model(input_ids= input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,labels=labels)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                logits= logits.detach().cpu().numpy()
                labels=labels.to('cpu').numpy()
                predictions=np.argmax(logits,axis=1)
                for item in predictions:
                    f.write(str(item)+'\n')
                tmp_eval_accuracy=np.sum(predictions==labels)
                source_eval_loss+=outputs[0].mean().item()
                source_eval_accuracy+=tmp_eval_accuracy
                nb_source_eval_examples+=input_ids.size(0)
                nb_source_eval_steps+=1
        eval_loss=eval_loss/nb_eval_steps
        eval_accuracy=eval_accuracy/nb_eval_examples
        source_eval_loss=source_eval_loss/nb_source_eval_steps
        source_eval_accuracy=source_eval_accuracy/nb_source_eval_examples
        if eval_accuracy>best_eval_accuracy:
            best_eval_accuracy=eval_accuracy
            logger.info("Saving trained model at {}...".format(train_args.output_dir))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(train_args.output_dir)
        result={
            'epoch': epoch,
            'eval_loss':eval_loss,
            'eval_accuracy':eval_accuracy,
            'source_eval_loss':source_eval_loss,
            'source_eval_accuracy':source_eval_accuracy,
            'best_eval_accuracy': best_eval_accuracy,
            'global_step':global_step,
            'loss':tr_loss/nb_tr_steps,
            'domain_loss':tr_domain_loss/nb_tr_steps,
            'consume_time(s)':end_time-start_time
        }
        output_eval_file=os.path.join(train_args.output_dir,"eval_results_ep"+str(epoch)+".txt")
        with open(output_eval_file,"w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
