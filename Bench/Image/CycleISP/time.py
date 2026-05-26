import torch
import time
from networks.denoising_rgb import DenoiseNet
from networks.resnet_LNP import resnet50           # 替换为你定义的 ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型1：去噪模型
denoise_model = DenoiseNet().to(device)
denoise_model.eval()

# 模型2：分类模型
classifier_model = resnet50(num_classes=1).to(device)
classifier_model.load_state_dict(torch.load("/data/lp/CNNDetection/checkpoints/retrain_LNPmodel/model_epoch_best.pth")["model"])
classifier_model.eval()

# 模拟10帧视频数据
input_tensor = torch.randn(10, 3, 224, 224).to(device)  # 修改尺寸为你的模型所需尺寸

with torch.no_grad():
    # 预热
    for _ in range(10):
        _ = classifier_model(denoise_model(input_tensor))

    torch.cuda.synchronize()
    start = time.time()
    repeat = 1000
    for _ in range(repeat):
        denoised = denoise_model(input_tensor)
        _ = classifier_model(denoised)
    torch.cuda.synchronize()
    end = time.time()

avg_time = (end - start) * 1000 / repeat  # 单次流水线平均时间（ms）
print(f"Average end-to-end inference time: {avg_time:.2f} ms for {input_tensor.shape[0]} frames")

from fvcore.nn import FlopCountAnalysis, parameter_count

denoise_input = torch.randn(1, 3, 224, 224).to(device)
classify_input = torch.randn(1, 3, 224, 224).to(device)  # 若前者输出同尺寸也可复用

# 参数统计
denoise_params = sum(p.numel() for p in denoise_model.parameters()) / 1e6
classify_params = sum(p.numel() for p in classifier_model.parameters()) / 1e6
print(f"Denoise Params: {denoise_params:.2f} M, Classifier Params: {classify_params:.2f} M")

# FLOPs
flops_denoise = FlopCountAnalysis(denoise_model, denoise_input).total()
flops_classify = FlopCountAnalysis(classifier_model, classify_input).total()
print(f"Denoise FLOPs: {flops_denoise / 1e9:.2f} GFLOPs, Classifier FLOPs: {flops_classify / 1e9:.2f} GFLOPs")

# 总 FLOPs
print(f"Total FLOPs: {(flops_denoise + flops_classify) / 1e9:.2f} GFLOPs")
