import torch
from torch import nn, einsum
import torch.nn.functional as F
import time
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import modules.pointnet2_utils as pointnet2_utils
import numpy as np
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    
    def forward(self, xyzs, feature):
        b,l,n, _, h = *feature.shape, self.heads
        device = xyzs.get_device()
        norm_features = self.norm(feature)
        q,k,v = self.to_qkv(norm_features).chunk(3, dim = -1)
        #q = map(lambda t: rearrange(t, 'b l n (h d) -> b l h n d', h = h), q)
        
        query_s = torch.split(tensor=q, split_size_or_sections=1, dim=1)  # 根据每个切片中的帧数进行分割  
        query_s= [torch.squeeze(input=q, dim=1).contiguous() for q in query_s]  

        xyzs1=rearrange(xyzs,'b l n d -> (b l) n d',b=b,l=l)#(c) n 3
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)  # 根据每个切片中的帧数进行分割  
        xyzs= [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]  
        keys_video=rearrange(k,'b l n d -> (b l) n d',b=b,l=l)#(c) n d
        values_video=rearrange(v,'b l n d -> (b l) n d',b=b,l=l)#(c) n d

        for i in range(0,l):
            res_att=[]
            dis=[]

            xyzs_expand=xyzs[i].unsqueeze(1).repeat(1,l,1,1) #b l n 3

            xyzs_expand=rearrange(xyzs_expand,'b l n d -> (b l) n d',b=b,l=l)#(c) n 3
            idx = pointnet2_utils.ball_query(0.2,8, xyzs1, xyzs_expand)#c n k
            key_filp=keys_video.transpose(1, 2).contiguous() #c d n
            temporal_neighbor_key_grouping=pointnet2_utils.grouping_operation(key_filp, idx)#c d n k
            
            values_filp=values_video.transpose(1, 2).contiguous()
            temporal_neighbor_value_grouping=pointnet2_utils.grouping_operation(values_filp, idx)#c d n k
            
            neighbor_xyz_filp = xyzs1.transpose(1, 2).contiguous()#c 3 n
            temporal_neighbor_xyz_grouping=pointnet2_utils.grouping_operation(neighbor_xyz_filp, idx)#c 3 n k
            spatial_temporal_displacements=temporal_neighbor_xyz_grouping.permute(0,2,3,1)-xyzs_expand.unsqueeze(2)  #c n k 3

            keys=rearrange(temporal_neighbor_key_grouping.permute(0,2,3,1),'(b l) n k (h d)->(b n) h (l k 1) d',b=b,l=l,h=h,k=8)
            values=rearrange(temporal_neighbor_value_grouping.permute(0,2,3,1),'(b l) n k (h d)->(b n) h (l k 1) d',b=b,l=l,h=h,k=8)
            query=rearrange(query_s[i],'b n (h d) -> (b n) h 1 d',h=h)#b n d


            attn_weight=torch.einsum('c h i d,c h j d->c h i j',query,keys)*self.scale #(b n) h i j 1

            attn = attn_weight.softmax(dim=-1)

            
            
            
            attn_dis=attn.unsqueeze(4)

            displacements=rearrange(spatial_temporal_displacements,'(b l) n k d->b l n k d',l=l,k=8)
            dis=rearrange(displacements,'b l n k d->(b n) (l k 1) d',l=l).unsqueeze(1).unsqueeze(2)
            dis_attn=attn_dis*dis#c h i j d
            
            dis_attn=torch.max(input=dis_attn,dim=2,keepdim=False)[0]
            
            dis_attn=torch.max(input=dis_attn,dim=2,keepdim=False)[0]#c h 3
            dis_attn=self.spatial_op(dis_attn)
            dis_attn=rearrange(dis_attn,'(b n) h d->b n (h d)',n=n,h=h)
            
            attnout=torch.einsum('c h i j,c h j d->c h i d',attn,values)
            attnout=rearrange(attnout,'(b n) h 1 d->b n (h 1 d)',n=n,h=h)
            attnout=attnout+dis_attn
            res_att.append(attnout)

        frame_result=torch.stack(res_att,dim=1)#b t n (h d)
        out =  self.to_out(frame_result)#b 1 d
        
        return out+feature
    
class LLAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.05):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = 0.),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features
        