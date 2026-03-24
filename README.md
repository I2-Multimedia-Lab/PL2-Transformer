# PL2-Transformer
🔥Official implement of "Point Long-Term Locality-Aware Transformer for Point Cloud Video Understanding" (ACM MM Asia Workshop on Imaging, Processing, Perception, and Reasoning for High-Dimensional Visual Data)
## Abstract
🔥Point cloud videos have been widely used in real-world applications to understand 3D dynamic objects and scenes. However, there still exist significant challenges in effectively embedding the inter-frame motion. Another crucial challenge lies in the failure to consider the long-term dependencies within local regions, which is an important factor for the efficacy of the neural model yet largely under-explored. In this paper, we propose an effective Point Long-term Locality-aware Transformer network to meet these challenges, termed as PL2-Transformer. First, the Point 4D Convolution (4DConv) is harnessed as the 4D backbone to aggregate the short-term spatial-temporal local information. Second, to enhance motion dynamics understanding, we introduce an inter-frame motion embedding, which captures the motion between frames and provides reliable motion cues for the subsequent Transformer network. Finally, we propose an effective Long-Term Locality-Aware Transformer (LLT), which utilizes a novel Long-Term Locality-Aware Attention (LLA) mechanism to capture long-term dependencies within local regions across the entire Point cloud video. Extensive experiments on multiple benchmarks demonstrate the effectiveness of our approach, surpassing the current state-of-the-art (SOTA) methods, or being comparable to the current SOTA methods while having fewer parameters. Source codes will be made publicly available. 
## Installation

The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 8.3.1, PyTorch v1.8.1, CUDA 10.2 and cuDNN v7.6.  
Device : 2 × RTX 2080Ti (22G)  
Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search: 
```
cd modules
python setup.py install
```
## Datasets
🌱The MSR dataset encompasses 20 actions, a total of 23K frames. Thanks to this author [Meteornet](https://github.com/xingyul/meteornet) for providing us with data preprocessing code.
(about 800M)  
🌱The NTU RGB+D 60 dataset encompasses 60 actions, a total of 4M frames. Thanks to this author [PSTNet](https://github.com/hehefan/Point-Spatio-Temporal-Convolution) for providing us with data preprocessing code.
(about 800G)  
🌱The Synthia 4D dataset. Synthia 4D is a synthetic dataset for outdoor autonomous driving. Thanks to this author [P4transformer](https://github.com/hehefan/P4Transformer) for providing us with data preprocessing code.
(about 5G)  
🌱The NVgesture dataset. Thanks to this author [MaST-Pre](https://github.com/JohnsonSign/MaST-Pre) for providing us with data preprocessing code.
(about 10G)  
## Train
🤗Lets train the model!
### Train medium model
```
python train-msr-meduim.py
```
### Train full model
```
python train-msr-full.py
```

## Log
📢The log has been uploaded!

## PIPELINE
![pipeline](https://github.com/I2-Multimedia-Lab/PL2-Transformer/blob/main/Pipeline.png)

## Inter-Frame Motion Visualization
![Motion](https://github.com/I2-Multimedia-Lab/PL2-Transformer/blob/main/Img/Fig5V2.png)

## Visualization of 4D Semantic Segmentation
![visualization](https://github.com/I2-Multimedia-Lab/PL2-Transformer/blob/main/experiments_synthia_visualizationV3.jpg)
## Related Repos

1. PointNet++ PyTorch implementation: https://github.com/facebookresearch/votenet/tree/master/pointnet2
2. Transformer: https://github.com/lucidrains/vit-pytorch
3. P4Transformer: [https://github.com/hehefan/P4Transformer](https://github.com/hehefan/P4Transformer)
4. PST-Transformer:https://github.com/hehefan/PST-Transformer

## Acknowledgement

💡We thank the authors of [P4transformer](https://github.com/hehefan/P4Transformer) and [PST-Transformer](https://github.com/hehefan/PST-Transformer) for their interesting work.

## 
@inproceedings{10.1145/3769748.3773341,
author = {Zuo, Zhi and Gao, Pan and Paul, Manoranjan},
title = {Point Long-Term Locality-Aware Transformer for Point Cloud Video Understanding},
year = {2025},
isbn = {9798400722479},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3769748.3773341},
doi = {10.1145/3769748.3773341},
abstract = {Point cloud videos are widely applied in understanding 3D dynamic objects and scenes. However, a key challenge lies in effectively modeling inter-frame motion and capturing long-term dependencies within local regions. To this end, we propose the Point Long-term Locality-aware Transformer (PL2-Transformer) to address these issues. A Point 4D Convolution (4DConv) backbone aggregates short-term spatiotemporal features, while an inter-frame motion embedding module explicitly models frame-to-frame motion. Furthermore, a Long-Term Locality-Aware Transformer (LLT) with a novel attention mechanism captures long-range dependencies across local regions. Experiments on multiple benchmarks show that our method achieves competitive or superior performance compared to state-of-the-art approaches. The code is available at https://github.com/I2-Multimedia-Lab/PL2-Transformer},
booktitle = {Proceedings of the 7th ACM International Conference on Multimedia in Asia},
articleno = {3},
numpages = {8},
keywords = {point cloud videos, long-term locality-aware transformer},
location = {
},
series = {MMAsia '25 Workshops}
}
