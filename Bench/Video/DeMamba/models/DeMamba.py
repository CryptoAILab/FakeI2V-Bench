from transformers import XCLIPVisionModel, AutoProcessor
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from mamba_base import MambaConfig, ResidualBlock
import torch.nn.init as init
from clip import clip
import math
import av
import time
from thop import profile


def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data

class XCLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=768, class_num=1
    ):
        super(XCLIP_DeMamba, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained("/data2/lp/Video-based/DeMamba/GenVideo/pretrained_weights/xclip")
        blocks = []
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear((self.patch_nums+1)*channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.initialize_weights(self.fc1)
        self.dropout = nn.Dropout(p=0.0)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        #print(x.shape)
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape
        #print(sequence_output.shape)

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        sequence_output = sequence_output.view(b, t, -1, c)
        #print(sequence_output.shape)

        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        #print(s)
        b_l = b*s*s*4
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        #print(x.shape)
        res = self.mamba(x)
        #print(res.shape)

        video_level_features = res.mean(1)
        #print(video_level_features.shape)
        video_level_features = video_level_features.view(b, -1)
        #print(video_level_features.shape)
        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(video_level_features)
        pred = self.dropout(pred)

        return pred



class CLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=512, class_num=1
    ):
        super(CLIP_DeMamba, self).__init__()
        self.clip_model, preprocess = clip.load('ViT-B-14')
        self.clip_model = self.clip_model.float()
        blocks = []
        channel = 512
        self.fusing_ratios = 2
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear(channel*(self.patch_nums+1), class_num)
        self.bn1 = nn.BatchNorm1d(channel)
        self.initialize_weights(self.fc1)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, -1, c)

        global_feat = sequence_output.reshape(b, -1, c)
        global_feat = global_feat.mean(1)

        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)
        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)

        video_level_features = torch.cat((global_feat, video_level_features), dim=1)
        x = self.fc1(video_level_features)

        return x

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
    
    file_path = '/data2/lp/IVBridge_datasets/SourceVideos/GenVideo/SourceVideos/Crafter/Crafter_1.mp4'
    container = av.open(file_path)

    indices = sample_frame_indices(clip_len = 8, frame_sample_rate = 1, seg_len = container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    processor = AutoProcessor.from_pretrained('microsoft/xclip-base-patch32')
    pixel_values = processor(videos = list(video), return_tensors = "pt").pixel_values
    print(pixel_values.shape)
    batch_size, num_frames, num_channels, height, width = pixel_values.shape
    # pixel_values = pixel_values.reshape(-1, num_channels, height, width)

    model = XCLIP_DeMamba()

    outputs= model(pixel_values)
    print(outputs)
    dummy_input = pixel_values
    # === 1. 参数量和 FLOPs ===
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    # === 2. 推理时间 ===
    with torch.no_grad():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        # 正式计时
        starter.record()
        for _ in range(100):
            _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        elapsed_time = starter.elapsed_time(ender) / 100  # 平均每次毫秒
        print(f"Inference Time: {elapsed_time:.3f} ms per image")