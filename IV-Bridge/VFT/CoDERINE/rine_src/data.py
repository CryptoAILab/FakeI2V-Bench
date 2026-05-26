from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from src.perturbation import perturbation


class TrainingDataset(Dataset):
    def __init__(self, split, classes=None, transforms=None, ds_frac=None):
        self.real = [
            (f"data/{split}/{y}/0_real/{x}", 0)
            for y in classes
            for x in os.listdir(f"data/{split}/{y}/0_real")
        ]
        self.fake = [
            (f"data/{split}/{y}/1_fake/{x}", 1)
            for y in classes
            for x in os.listdir(f"data/{split}/{y}/1_fake")
        ]

        self.images = self.real + self.fake
        random.shuffle(self.images)
        if ds_frac is not None:
            self.images = self.images[: int(len(self.images) * ds_frac)]

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, target]


class TrainingDatasetLDM(Dataset):
    def __init__(self, split, transforms=None):
        self.real = [
            (f"data/train/{x.split('_')[0]}/0_real/{x.split('_')[1]}", 0)
            for x in pd.read_csv(
                f"data/latent_diffusion_trainingset/{split}/real_lsun.txt",
                header=None,
            )
            .values.reshape(-1)
            .tolist()
        ] + [
            (
                (
                    f"data/coco/train2014/COCO_train2014_{x}"
                    if os.path.exists(f"data/coco/train2014/COCO_train2014_{x}")
                    else f"data/coco/val2014/COCO_val2014_{x}"
                ),
                0,
            )
            for x in pd.read_csv(
                f"data/latent_diffusion_trainingset/{split}/real_coco.txt", header=None
            )
            .values.reshape(-1)
            .tolist()
        ]
        fake_dir = "data/latent_diffusion_trainingset/"
        self.fake = [
            (f"{fake_dir}{split}/{x}/{y}", 1)
            for x in os.listdir(f"{fake_dir}{split}")
            if os.path.isdir(f"{fake_dir}{split}/{x}")
            for y in os.listdir(f"{fake_dir}{split}/{x}")
        ]
        self.images = self.real + self.fake
        random.shuffle(self.images)

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, target]


class EvaluationDataset(Dataset):
    def __init__(self, generator, transforms=None, perturb=None, max_sample=32):
        if (generator in [
            "Celeb-synthesis_frames",
            "Crafter/Crafter_frames",
                "Gen2/Gen2_frames","HotShot/HotShot_frames",
                "Lavie/Lavie_frames","ModelScope/ModelScope_frames",
                "MoonValley/MoonValley_frames","MorphStudio/MorphStudio_frames",
                "Show_1/Show_1_frames","Sora/Sora_frames",
                "WildScrape/WildScrape_frames"
                ]):
            self.fake = [
                (f"/data/lp/GenVideo/SourceFrames/1_fake/{generator}/{y}/{x}", 1)
                for y in os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/{generator}")
                for x in random.sample(os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/{generator}/{y}"),
                                       min(max_sample, 
                                            len(os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/{generator}/{y}"))))
            ]
            self.real = [
                (f"/data/lp/GenVideo/SourceFrames/0_real/toy/{y}/{x}", 0)
                for y in os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/toy")
                for x in random.sample(os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/toy/{y}"),
                                       min(max_sample,
                                           len(os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/toy/{y}"))))
            ]
        elif (generator in ["Celeb-real_frames", "MSRVTT/MSRVTT_frames"]):
            self.real = [
                (f"/data/lp/GenVideo/SourceFrames/0_real/{generator}/{y}/{x}", 0)
                for y in os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/{generator}")
                for x in random.sample(os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/{generator}/{y}"), 
                                        min(max_sample, 
                                            len(os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/{generator}/{y}"))))
            ]
            self.fake = [
                (f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames/{y}/{x}", 1)
                for y in os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames")
                for x in random.sample(os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames/{y}"))))
            ]
        elif (generator in ["DFD", "FaceShifter"]):
            self.real = [
                (f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames/{y}/{x}", 0)
                for y in os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames")
                for x in random.sample(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames/{y}"))))
            ]
            self.fake = [
                (f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames/{y}/{x}", 1)
                for y in os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames")
                for x in random.sample(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames/{y}"))))
            ]
        elif (generator in ["CelebDFV2"]):
            self.real = [
                (f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames/{y}/{x}", 0)
                for y in os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames")
                for x in random.sample(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/real/Frames/{y}"))))
            ]
            self.fake = [
                (f"/data2/lp/FacialDatasets/SourceFrames/{generator}/Celeb-fake/Frames/{y}/{x}", 1)
                for y in os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames")
                for x in random.sample(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data2/lp/FacialDatasets/SourceFrames/{generator}/fake/Frames/{y}"))))
            ]        
        elif (generator in ["cogvideo", "musev", "mora", "svd"]):
            self.real = [
                (f"/data/lp/GenVideo/SourceFrames/0_real/toy/{y}/{x}", 0)
                for y in os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/toy")
                for x in random.sample(os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/toy/{y}"),
                                       min(max_sample,
                                           len(os.listdir(f"/data/lp/GenVideo/SourceFrames/0_real/toy/{y}"))))
            ]
            self.fake = [
                (f"/data2/lp/GenVidBench/SourceFrames/fake/{generator}/Frames/{y}/{x}", 1)
                for y in os.listdir(f"/data2/lp/GenVidBench/SourceFrames/fake/{generator}/Frames")
                for x in random.sample(os.listdir(f"/data2/lp/GenVidBench/SourceFrames/fake/{generator}/Frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data2/lp/GenVidBench/SourceFrames/fake/{generator}/Frames/{y}"))))
            ]
        elif (generator in ["Vript"]):
            self.real = [
                (f"/data2/lp/GenVidBench/SourceFrames/real/{generator}/Frames/{y}/{x}", 0)
                for y in os.listdir(f"/data2/lp/GenVidBench/SourceFrames/real/{generator}/Frames")
                for x in random.sample(os.listdir(f"/data2/lp/GenVidBench/SourceFrames/real/{generator}/Frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data2/lp/GenVidBench/SourceFrames/real/{generator}/Frames/{y}"))))
            ]
            self.fake = [
                (f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames/{y}/{x}", 1)
                for y in os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames")
                for x in random.sample(os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames/{y}"), 
                                    min(max_sample, 
                                        len(os.listdir(f"/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames/{y}"))))               
            ]


        
        # if generator in ["cyclegan", "progan", "stylegan", "stylegan2"]:
        #     self.real = [
        #         (f"data/test/{generator}/{y}/0_real/{x}", 0)
        #         for y in os.listdir(f"data/test/{generator}")
        #         for x in os.listdir(f"data/test/{generator}/{y}/0_real")
        #     ]
        #     self.fake = [
        #         (f"data/test/{generator}/{y}/1_fake/{x}", 1)
        #         for y in os.listdir(f"data/test/{generator}")
        #         for x in os.listdir(f"data/test/{generator}/{y}/1_fake")
        #     ]
        # elif "diffusion_datasets/guided" in generator:
        #     self.real = [
        #         (f"data/test/diffusion_datasets/imagenet/0_real/{x}", 0)
        #         for x in os.listdir(f"data/test/diffusion_datasets/imagenet/0_real")
        #     ]
        #     self.fake = [
        #         (f"data/test/{generator}/1_fake/{x}", 1)
        #         for x in os.listdir(f"data/test/{generator}/1_fake")
        #     ]
        # elif (
        #     "diffusion_datasets/ldm" in generator
        #     or "diffusion_datasets/glide" in generator
        #     or "diffusion_datasets/dalle" in generator
        # ):
        #     self.real = [
        #         (f"data/test/diffusion_datasets/laion/0_real/{x}", 0)
        #         for x in os.listdir(f"data/test/diffusion_datasets/laion/0_real")
        #     ]
        #     self.fake = [
        #         (f"data/test/{generator}/1_fake/{x}", 1)
        #         for x in os.listdir(f"data/test/{generator}/1_fake")
        #     ]
        # elif any(
        #     [
        #         x in generator
        #         for x in [
        #             "biggan",
        #             "stargan",
        #             "gaugan",
        #             "deepfake",
        #             "seeingdark",
        #             "san",
        #             "crn",
        #             "imle",
        #         ]
        #     ]
        # ):
        #     self.real = [
        #         (f"data/test/{generator}/0_real/{x}", 0)
        #         for x in os.listdir(f"data/test/{generator}/0_real")
        #     ]
        #     self.fake = [
        #         (f"data/test/{generator}/1_fake/{x}", 1)
        #         for x in os.listdir(f"data/test/{generator}/1_fake")
        #     ]
        # elif any( #synthbuster dataset
        #     [
        #         x in generator
        #         for x in [
        #             "dalle2",
        #             "dalle3",
        #             "stable-diffusion-1-3",
        #             "stable-diffusion-1-4",
        #             "stable-diffusion-2",
        #             "stable-diffusion-xl",
        #             "glide",
        #             "firefly",
        #             "midjourney-v5",
        #         ]
        #     ]
        # ):
        #     self.real = [(f"data/RAISEpng/{x}", 0) for x in os.listdir("data/RAISEpng")]
        #     self.fake = [
        #         (f"data/synthbuster/{generator}/{x}", 1)
        #         for x in os.listdir(f"data/synthbuster/{generator}")
        #         if all([y not in x for y in [".txt", ".py"]])
        #     ]

        self.images = self.real + self.fake

        self.transforms = transforms
        self.perturb = perturb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        video_name = image_path.split("/")[-2]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None and self.perturb is None:
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)
            else:
                image = self.transforms(image)
        return [image, target, video_name]
