import torch
from torch import nn
from torchvision.models import resnet18

import numpy as np
import pandas as pd
from utils.memory import log_mem


model=resnet18().cuda()
batch_size=128
input=torch.rand(batch_size,3,224,224).cuda()

mem_log=[]


mem_log.extend(log_mem(model,input,mem_log))

for idx,item in enumerate(mem_log):
    print(f'idx={idx}\n item={item}\n')



import matplotlib.pyplot as plt

def memplot(df):
    fig,ax=plt.subplots(figsize=(20,10))
    pl=df.plot(y='mem_all')
    pl.get_figure().savefig('./mem_all.png')
    return

df=pd.DataFrame(mem_log)
memplot(df)

