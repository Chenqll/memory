import torch
from torch import nn
from torchvision.models import resnet18

import numpy as np
import pandas as pd
from utils.memory import log_mem
from utils.plot_mem_change import memplot


model=resnet18().cuda()
batch_size=128
input=torch.rand(batch_size,3,224,224).cuda()

mem_log=[]

# can draw pre,fwd,bwd or all
draw_type='' 

mem_log.extend(log_mem(model,input,mem_log))

for idx,item in enumerate(mem_log):
    print(f'idx={idx}\n item={item}\n')


df=pd.DataFrame(mem_log)
memplot(df,draw_type)

