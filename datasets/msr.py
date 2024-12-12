import os
import sys
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data
from torch.utils.data.dataloader import default_collate


def clip_normalize(clip):
    # Reshape the input clip to a point cloud with shape (n_points, 3)
    pc = np.reshape(a=clip, newshape=[-1, 3])
    
    # Calculate the maximum and minimum values along each dimension
    pc_max = pc.max(axis=0)  # Shape (3,)
    pc_min = pc.min(axis=0)  # Shape (3,)
    
    # Compute the center and the scale for normalization
    shift = (pc_min + pc_max) / 2  # Shape (3,)
    scale = (pc_max - pc_min).max() / 2  # Scalar value
    
    # Normalize the point cloud
    pc = (pc - shift) / scale
    
    # Reshape back to the original clip shape
    clip1 = np.reshape(a=pc, newshape=[1, -1, 3])
    
    return clip1, shift, scale



class MSRAction3D(Dataset):
    def __init__(self, root, frames_per_clip=8, frame_interval=1, num_points=2048, train=True):
        super(MSRAction3D, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                #print(video.shape)
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-frame_interval*(frames_per_clip-1)):
                    self.index_map.append((index, t))
                index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]

        label = self.labels[index]
        
        clip = [video[t+i*self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        #clip,shift,scale = clip_normalize(np.array(clip))
        #print
        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300

        return clip.astype(np.float32), label, index

if __name__ == '__main__':
    dataset = MSRAction3D(root='/home/zz/P4Transformer-ours/data/msr_action', frames_per_clip=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=14, shuffle=True, pin_memory=True)
    print(len(dataset))
    clip, label, video_idx,shift,scale = dataset[373]
    np.save("clip.npy",clip)
    print(clip.shape)
    print(label)
    print(video_idx)
    print(dataset.num_classes)
