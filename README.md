# FakeI2V-Bench



## Datasets

The following datasets are used in our benchmark:

- **CelebDF-V2**
- **DeepFakeDetection**
- **GenVideo**
- **GenVidBench**

They are all open-source and can be publicly downloaded.

## Detection Methods

We evaluated the following detectors:

- **Image-level detectors**: 12 methods (original code available CNNDet, LNP, Patch, LGrad, DIRE, DeFake, CoDE, RINE, UniFD, DRCT, DMID, NPR)  
- **Video-level detectors**: 8 methods (original code available FTCN, AltFreezing, UIA-ViT, LAA-Net, DeMamba, VGMShield, MM-Det, D3 )


## Directory Structure
```
FakeI2VBench
├── Bench
│ ├── Image-level
│ │ ├── CNNDet
│ │ ├── DeFake
│ │ └── ...
│ └── Video-level
│ ├── D3
│ ├── VGMshield
│ └── ...
│
├── IV-Bridge
│ ├── VFT
│ └── MMA
```

### Description

- **Bench/**  
  Contains code of deepfake detectors on four datasets.  
  - **Image-level/** — Code for 12 image-level detectors, e.g., `CNNDet`, `DeFake`, etc.  
  - **Video-level/** — Code for 8 video-level detectors, e.g., `D3`, `VGMshield`, etc.  

- **IV-Bridge/**  
  Contains methods for improving detection performance.  
  - **VFT/** — Used to train 12 image-level detectors.  
  - **MMA/** — Implements multi-mode aggregation using Random Forest.

