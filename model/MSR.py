import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
from modules.point_4d_convolution import *
from modules.Inter_Frame_Motion import *
from modules.LLA_MSR import *
class PL2Transformer(nn.Module):
    def __init__(self, spatial_stride,emb_relu,radius, nsamples ,                            
                 dim,depth, heads, dim_head,  mlp_dim,                                             
                 num_classes):                                                 
        super().__init__()

        
        self.inter_frame_motion=IFM(in_planes=512,
                                            temporal_stride=1,
                                            spatial_stride=spatial_stride*2,
                                            mlp_planes=[1024],
                                            spatial_kernel_size=[0.5,16],
                                            emb_relu=emb_relu,
                                            temporal_padding=[1,1])
        
        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[512], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=3, temporal_stride=2, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.long2transformer = LLAttention(1024, depth, heads, dim_head, mlp_dim,dropout=0.)

        self.emb_relu = nn.ReLU() if emb_relu else False
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(mlp_dim, num_classes),
        )

        
    def forward(self, input):              

        
        xyzs1,features1=self.tube_embedding(input)
        
        features1=self.emb_relu(features1).permute(0,1,3,2)
        
        xyzs,features4=self.inter_frame_motion(xyzs1,features1)
        
        output = self.long2transformer(xyzs, features4)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
    
        return output
