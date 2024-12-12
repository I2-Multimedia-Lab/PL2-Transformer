
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
import modules.pointnet2_utils as pointnet2_utils
from typing import List
class FlowEmbedding(nn.Module):  
    def __init__(self, 
                 in_planes: int,
                 spatial_kernel_size: [float, int],
                 mlp_planes: List[int],
                 ):          
        super(FlowEmbedding, self).__init__()  
        self.in_planes=in_planes
        self.r,self.k=spatial_kernel_size
        
        self.convf=nn.Sequential(nn.Conv2d(in_channels=in_planes+3,out_channels=mlp_planes[0],kernel_size=1,bias=False),nn.ReLU(inplace=True),nn.BatchNorm2d(num_features=mlp_planes[0]))
        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=False))
            #if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            #if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)
    def forward(self, anchor_xyzs, neighbor,neighbor_features,features):  
        start_time=time.time()
        batch_size,npoints,ndims = anchor_xyzs.size()   
        device = anchor_xyzs.get_device()  
        
        """
            input   
                neighbor:(batch_size,n/s1,3)
                anchor_xyzs:(batch_size,n/s2,3)
                neighbor_features:(batch_size,n/s1,dim)
                features:(batch_size,n/s1,dim)
        output:
                anchor_xyz:(batch_size,n/s2,3)
                feature_s:(batch_size,n/s2,outchannel)
                
        """

        idx = pointnet2_utils.ball_query(self.r, self.k, neighbor, anchor_xyzs) #索引
        neighbor_flipped = neighbor.transpose(1, 2).contiguous()                                                    # (B, 3, N)
        neighbor_grouped = pointnet2_utils.grouping_operation(neighbor_flipped, idx) #b 3 n/s K
        neighbor_query_ball=neighbor_grouped.permute(0,2,3,1) #b n/s k 3
        y = anchor_xyzs.view(batch_size, npoints, 1, ndims).repeat(1, 1, self.k, 1)
        xyz_dis=y-neighbor_query_ball

        features=features.transpose(1, 2).contiguous()
        neighbor_features=neighbor_features.transpose(1, 2).contiguous()
        features_Q=pointnet2_utils.grouping_operation(features , idx).permute(0,2,3,1).contiguous()    #b dim n/s2 k ->b n/s2 k dim 
        features_KV=pointnet2_utils.grouping_operation(neighbor_features , idx).permute(0,2,3,1).contiguous()   #b dim n/s2 k ->b n/s2 k dim
        feature_dis=features_Q*features_KV
        feature_new=torch.cat((xyz_dis,feature_dis),dim=-1)#batchsize,npoints,inplanes+3
        out=self.convf(feature_new.permute(0,3,1,2))
        out=self.mlp(out)

        out=torch.max(input=out,dim=-1,keepdim=False)[0].permute(0,2,1)

        return out
        