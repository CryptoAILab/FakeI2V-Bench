import torch
import torch.nn as nn
import clip


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return p, z
        # return p
# class Model(nn.Module):
#     def __init__(
#         self,
#         backbone,
#         nproj,
#         proj_dim,
#         device,
#     ):
#         super().__init__()

#         self.device = device

#         # Load and freeze CLIP
#         self.clip, self.preprocess = clip.load(backbone[0], device=device)
#         for name, param in self.clip.named_parameters():
#             param.requires_grad = False

#         # Register hooks to get intermediate layer outputs
#         self.hooks = [
#             Hook(name, module)
#             for name, module in self.clip.visual.named_modules()
#             if "ln_2" in name
#         ]

#         # Initialize the trainable part of the model
#         self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
#         proj1_layers = [nn.Dropout()]
#         for i in range(nproj):
#             proj1_layers.extend(
#                 [
#                     nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
#                     nn.ReLU(),
#                     nn.Dropout(),
#                 ]
#             )
#         self.proj1 = nn.Sequential(*proj1_layers)

#         proj2_layers = [nn.Dropout()]
#         for _ in range(nproj):
#             proj2_layers.extend(
#                 [
#                     nn.Linear(proj_dim, proj_dim),
#                     nn.ReLU(),
#                     nn.Dropout(),
#                 ]
#             )
#         self.proj2 = nn.Sequential(*proj2_layers)

#         self.head = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(proj_dim, 1),
#         )

#     def forward(self, x, for_attack: bool = False):
#         # CLIP 权重依旧冻结，不需要对参数求梯度
#         if for_attack:
#             _ = self.clip.encode_image(x)  # 对抗攻击时允许梯度
#         else:
#             with torch.no_grad():
#                 _ = self.clip.encode_image(x)

#         # hooks 里存的是每一层的输出（含 batch 维）
#         g = torch.stack([h.output for h in self.hooks], dim=2)  # [T, B, L, C]

#         # 如果只取第0个 token，保留第一维
#         g = g[0:1, :, :, :]  # [1, B, L, C]
#         g = g.squeeze(0)
#         g = self.proj1(g.float())

#         z = torch.softmax(self.alpha, dim=1) * g
#         z = torch.sum(z, dim=1)
#         z = self.proj2(z)

#         p = self.head(z)  # [B, 1]

#         return p, z
