from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )

    def pca_residual(self, features, k = 256):
        X = features.T
        mean = X.mean(dim=1, keepdim= True)
        X_centered= X - mean

        U, S, Vh = torch.linalg.svd(X_centered, full_matrices = False)
        U_k = U[:, :k]
        X_proj = U_k@U_k.T@X_centered
        X_res = X_centered - X_proj

        residuals = X_res.T
        energy_ratio = (S[:k]**2).sum()/(S**2).sum()
    
        return residuals, U_k, energy_ratio
    

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

