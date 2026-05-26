
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
import torchvision.models as models

Transformers = [
    'CLIP-16',
    'CLIP-32',
    'XCLIP-16',
    'XCLIP-32',
    'DINO-base',
    'DINO-large',
]

class D3_model(nn.Module):
    def __init__(self, encoder_type = 'CLIP-16', loss_type = 'cos'):
        super(D3_model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type

        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        elif encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")

        elif encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

        elif encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")

        elif encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")

        elif encoder_type == 'ResNet-18':
            resnet18 = models.resnet18(pretrained=True)
            modules = list(resnet18.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'VGG-16':
            vgg16 = models.vgg16(pretrained=True)
            modules = list(vgg16.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'EfficientNet-b4':
            efficientnet_b4 = models.efficientnet_b4(pretrained=True)
            modules = list(efficientnet_b4.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval() 

        elif encoder_type == 'MobileNet-v3':
            mobilenetv3 = timm.create_model('mobilenetv3_large_100', pretrained=True)
            modules = list(mobilenetv3.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval() 

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)
        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            outputs = outputs.pooler_output
        else:
            outputs = self.encoder(images)
        outputs=outputs.reshape(b, t, -1)
        vec1 = outputs[:, :-1, :]  # [b, n-1, 768]
        vec2 = outputs[:, 1:, :]   # [b, n-1, 768]
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # [b, n-1]
        elif self.loss_type == 'l2':
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)  # [b, n-1]
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # [b, n-2]
        dis_2nd_avg = torch.mean(dis_2nd,dim=1)
        dis_2nd_std = torch.std(dis_2nd, dim=1) # [b]
        return outputs, dis_2nd_avg, dis_2nd_std