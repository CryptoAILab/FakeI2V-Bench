from transformers import XCLIPVisionModel, AutoProcessor
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import av
from sklearn.manifold import TSNE

from transformers import XCLIPVisionModel
class XCLIP(nn.Module):
    def __init__(
        self, channel_size=512, dropout=0.2, class_num=1
    ):
        super(XCLIP, self).__init__()
      
        self.backbone = XCLIPVisionModel.from_pretrained("GenVideo/pretrained_weights/xclip")
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.backbone(images, output_hidden_states=True)
        sequence_output = outputs['pooler_output'].reshape(b, t, -1)
        video_level_features = self.fc_norm(sequence_output.mean(1))
        pred = self.head(video_level_features)

        # return pred
        return pred, video_level_features

if __name__ == '__main__':
    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video = 0)):
            if i >end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format = "rgb24") for x in frames])
    
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num= clip_len)
        indices = np.clip(indices, start_idx, end_idx -1).astype(np.int64)
        return indices
    
    file_path = '/data2/lp/DeMamba/GenVideo/fake-video/Gen2/gen2_1.mp4'
    container = av.open(file_path)

    indices = sample_frame_indices(clip_len = 8, frame_sample_rate = 1, seg_len = container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    processor = AutoProcessor.from_pretrained('microsoft/xclip-base-patch32')
    pixel_values = processor(videos = list(video), return_tensors = "pt").pixel_values
    print(pixel_values.shape)
    batch_size, num_frames, num_channels, height, width = pixel_values.shape

    # pixel_values = pixel_values.reshape(-1, num_channels, height, width)

    model = XCLIP()

    outputs, features = model(pixel_values)
    print(features.type)
    features = features.cpu().detach().numpy()
    print(features.shape)

    # tsne = TSNE(n_components=2, random_state = 42)

    # X_tsne = tsne.fit_transform(features)
    # print(X_tsne.shape)








