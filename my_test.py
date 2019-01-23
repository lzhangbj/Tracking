import torch
import numpy as np

x = torch.load('rfcn_detect.pth')
# print(x['epoch'])
# print(x['session'])
# print(x['class_agnostic'])
print(x['model']['RFCN_cls_net.weight'].size())

