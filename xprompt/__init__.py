import os
import torch
import torch.nn as nn
import yaml
import sys
import logging.config
from typing import *

sys.path.append(".")

projectdir=os.path.abspath(os.getcwd())
cfgname=os.path.join(projectdir,'logger.yaml')
f=open(cfgname,"r",encoding='utf-8')
cfg=yaml.load(f,Loader=yaml.Loader)
f.close()
logging.config.dictConfig(cfg)
logger=logging.getLogger('simple')

__all__=['prompt_base','soft_prompt']