import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from typing import List


from modules.FlowEmbedding import *

class IFM(nn.Module):  
    def __init__(self, spatial_stride,emb_relu,
                 in_planes: int,
                 temporal_stride,
                 mlp_planes: List[int],
                 spatial_kernel_size: [float, int],
                 temporal_padding: [int, int] = [0, 0], 
                 temporal_padding_mode: str = 'replicate',):          
        super(IFM, self).__init__()  
        self.in_planes = in_planes
        self.temporal_stride=temporal_stride
        self.spatial_stride=spatial_stride
        self.temporal_padding_mode=temporal_padding_mode
        self.temporal_padding = temporal_padding
        self.emb_relu = nn.ReLU() if emb_relu else False
        self.r, self.k = spatial_kernel_size
        self.em=nn.ReLU()

        self.temporal_point_transformer_with_feature=FlowEmbedding(in_planes,[self.r,self.k],mlp_planes)
        
    def forward(self, xyzs, features=None):  
        
        npoint1 = xyzs.size(2) 
        device = xyzs.get_device()  
        
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)  
        xyzs= [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]  
        
        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]


        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyz=[]
        new_feature_temporal=[]
       
        for i in range(1,len(xyzs)-1,self.temporal_stride): 
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[i], npoint1//self.spatial_stride) #FPS降采样  # (B, N//self.spatial_stride)
            anchor_xyz = pointnet2_utils.gather_operation(xyzs[i].transpose(1, 2).contiguous(), anchor_idx).transpose(1, 2).contiguous() # (B, N//spatial_stride, 3)
            
            
            new_temporal=self.temporal_point_transformer_with_feature(anchor_xyz,xyzs[i+1],features[i+1],features[i])
                    
            new_feature_temporal.append(new_temporal)


            new_xyz.append(anchor_xyz) #b nframes n/s 3
        new_xyzs=torch.stack(tensors=new_xyz,dim=1) #b nframe n/s 3
        
        new_features_temporal=torch.stack(tensors=new_feature_temporal,dim=1) #b nframe,n c
        
        new_features=new_features_temporal

        return new_xyzs,new_features
        
        
        


 