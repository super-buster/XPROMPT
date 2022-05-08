from abc import ABC
from typing import List, Optional
from dataclasses import dataclass,field
import os

@dataclass
class TrainConfig:
    """Configuration for training a model
        param device: the device to use ('cpu' or 'gpu')
        param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        param n_gpu: the number of gpus to use
        param num_train_epochs: the number of epochs to train for
        param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        param weight_decay: the weight decay to use
        param learning_rate: the maximum learning rate to use
        param adam_epsilon: the epsilon value for Adam
        param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        param max_grad_norm: the maximum norm for the gradient
        param alpha: the alpha parameter for auxiliary language modeling
    """

    devices: Optional[List[int]] =field(
        default_factory=lambda:[0],
        metadata={"help":"choose gpu ids"}
    )
    task: str = field(
        default='rte',
        metadata={"help":"choose a finetune task from 'mnli','qqp','qnli','sst','cola','stsb','mrpc','rte','snli' "}
    )
    model_type: str =field(
        default= 'bert-base-cased',
        metadata={"help":"choose a model for finetuing"}
    )
    n_gpu : Optional[int] = field(
        default=1,
        metadata={"help":"num of gpu, -1 represents cpu"}
    )
    dataload_num_workers:Optional[int] = field(
        default=4,
        metadata={"help":"num of workers to load dataset"}
    )
    per_gpu_train_batch_size : Optional[int]= field(
        default= 32,
        metadata={"help":"training batch size per gpu"}
    ) 
    per_gpu_eval_batch_size : Optional[int]= field(
        default= 32,
        metadata={"help":"training batch size per gpu"}
    )          
    num_train_epochs :int =field(
        default=5
    )
    max_steps :Optional[int] =field(
        default=50000
    )
    max_seqlen :Optional[int] =field(
        default=256
    )
    gradient_accumulation_steps : Optional[int] =field(
        default=1
    )
    weight_decay : Optional[float] = field(
        default=0.01
    )
    adam_epsilon : Optional[float] =field(
        default=1e-6
    )
    warmup_steps : Optional[int] = field(
        default=0
    )
    max_grad_norm : Optional[float] =field(
        default=1.0
    )
    alpha : Optional[float] = field(
        default=0.65
    )
    learning_rate :float = field(
        default=2e-5
    )
    seed: int =field(
        default=42
    )
    output_dir: Optional[str] = field(
        default="/tmp/xprompt"
    )
    overwrite_output_dir: bool= field(
        default=True
    )
    resume: bool= field(
        default=False
    )




@dataclass
class EvalConfig:
    """Configuration for evaluating a model
        Create a new evaluation config.
        param device: the device to use ('cpu' or 'gpu')
        param n_gpu: the number of gpus to use
        param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        param metrics: the evaluation metrics to use (default: accuracy only)
        """
    devices: Optional[List[int]] =field(
            default_factory=lambda:[0],
            metadata={"help":"choose gpu ids"}
        )
    n_gpu : Optional[int] = field(
            default=-1,
            metadata={"help":"num of gpu, -1 represents cpu"}
    )
    per_gpu_eval_batch_size : Optional[int]= field(
            default= 32,
            metadata={"help":"infering batch size per gpu"}
    )   
    metrics : List[str] = field(
        default_factory=lambda:["acc"],
        metadata={"help":"evaluation metrics"}
    )
