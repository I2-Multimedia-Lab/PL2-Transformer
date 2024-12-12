# PL2-Transformer
🔥Official implement of "Point Long-Term Locality-Aware Transformer for Point Cloud Video Understanding" (Submitted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT))
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
## Core codes will coming soon.
## Datasets
🌱MSR Thanks to this author (https://github.com/xingyul/meteornet) for providing us with data preprocessing code.

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

## Acknowledgement

💡We thank the authors of [P4transformer](https://github.com/hehefan/P4Transformer) for their interesting work.
