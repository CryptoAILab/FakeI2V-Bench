DATASET_PATHS_IVB = [
    # ############################# GenVidBench dataset #########################################
    # dict(
    #     real_path='/data2/lp/IVBridge_datasets/crop/test/split/0real',
    #     fake_path='/data2/lp/IVBridge_datasets/crop/test/split/1fake',
    #     prompts_real='/data/lp/IVBridge-version3/DeFake/captions/train0/train0_real.txt',
    #     prompts_fake='/data/lp/IVBridge-version3/DeFake/captions/train0/train0_fake.txt',
    #     data_mode='wang2020',
    #     key='split'
    # ), 
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/CDFV2/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/CDFV2/1fake',
        prompts_real='/data/lp/IVBridge-version3/DeFake/captions/CDFV2/CDFV2_real.txt',
        prompts_fake='/data/lp/IVBridge-version3/DeFake/captions/CDFV2/CDFV2_fake.txt',
        data_mode='wang2020',
        key='CDFV2'
    ), 
        dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/DFD/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/DFD/1fake',
        prompts_real='/data/lp/IVBridge-version3/DeFake/captions/DFD/DFD_real.txt',
        prompts_fake='/data/lp/IVBridge-version3/DeFake/captions/DFD/DFD_fake.txt',
        data_mode='wang2020',
        key='DFD'
    ), 
        dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/GenVideo/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/GenVideo/1fake',
        prompts_real='/data/lp/IVBridge-version3/DeFake/captions/GenVideo/GenVideo_real.txt',
        prompts_fake='/data/lp/IVBridge-version3/DeFake/captions/GenVideo/GenVideo_fake.txt',
        data_mode='wang2020',
        key='GenVideo'
    ),
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/GVB/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/GVB/1fake',
        prompts_real='/data/lp/IVBridge-version3/DeFake/captions/GVB/GVB_real.txt',
        prompts_fake='/data/lp/IVBridge-version3/DeFake/captions/GVB/GVB_fake.txt',
        data_mode='wang2020',
        key='GVB'
    )
]