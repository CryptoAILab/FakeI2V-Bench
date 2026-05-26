DATASET_PATHS_D3 = [
    ############################# GneVideo dataset #########################################
        dict(
        real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
        fake_path='/data2/lp/GenVidBench/SourceFrames/fake/cogvideo/Frames',
        data_mode='wang2020',
        key='cogvideo'
    ),
    dict(
        real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
        fake_path='/data2/lp/GenVidBench/SourceFrames/fake/mora/Frames',
        data_mode='wang2020',
        key='mora'
    ),
    dict(
        real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
        fake_path='/data2/lp/GenVidBench/SourceFrames/fake/musev/Frames',
        data_mode='wang2020',
        key='musev'
    ),
    dict(
        real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
        fake_path='/data2/lp/GenVidBench/SourceFrames/fake/svd/Frames',
        data_mode='wang2020',
        key='svd'
    ),
    dict(
        real_path='/data2/lp/GenVidBench/SourceFrames/real/Vript/Frames',
        fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames',
        data_mode='wang2020',
        key='vript'
    ),
    dict(
        real_path='/data2/lp/FacialDatasets/SourceFrames/DFD/real/Frames',
        fake_path='/data2/lp/FacialDatasets/SourceFrames/DFD/fake/Frames',
        data_mode='wang2020',
        key='DFD'
    ),
    dict(
        real_path='/data2/lp/FacialDatasets/SourceFrames/FaceShifter/real/Frames',
        fake_path='/data2/lp/FacialDatasets/SourceFrames/FaceShifter/fake/Frames',
        data_mode='wang2020',
        key='FSh'
    ),
    dict(
        real_path='/data2/lp/FacialDatasets/SourceFrames/CelebDFV2/real/Frames',
        fake_path='/data2/lp/FacialDatasets/SourceFrames/CelebDFV2/Celeb-fake/Frames',
        data_mode='wang2020',
        key='Celeb-DFV2'
    ),
    # dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/Celeb-real_frames',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Celeb-synthesis_frames',
    #     data_mode='wang2020',
    #     key='CelebReal'
    # ),

    # dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/MSRVTT',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames',
    #     data_mode='wang2020',
    #     key='MSRVTT'
    # ),

    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/Celeb-real_frames',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Celeb-synthesis_frames',
    #     data_mode='wang2020',
    #     key='CelebDF'
    # ),


    # dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/ModelScope/ModelScope_frames',
    #     data_mode='wang2020',
    #     key='ModelScope'
    # ),
    #  dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/MorphStudio/MorphStudio_frames',
    #     data_mode='wang2020',
    #     key='MorphStudio'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/MoonValley/MoonValley_frames',
    #     data_mode='wang2020',
    #     key='MoonValley'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/HotShot/HotShot_frames',
    #     data_mode='wang2020',
    #     key='HotShot'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Show_1/Show_1_frames',
    #     data_mode='wang2020',
    #     key='Show_1'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Gen2/Gen2_frames',
    #     data_mode='wang2020',
    #     key='Gen2'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Crafter/Crafter_frames',
    #     data_mode='wang2020',
    #     key='Crafter'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Lavie/Lavie_frames',
    #     data_mode='wang2020',
    #     key='Lavie'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/Sora/Sora_frames',
    #     data_mode='wang2020',
    #     key='Sora'
    # ),
    #     dict(
    #     real_path='/data/lp/GenVideo/SourceFrames/0_real/toy',
    #     fake_path='/data/lp/GenVideo/SourceFrames/1_fake/WildScrape/WildScrape_frames',
    #     data_mode='wang2020',
    #     key='WildScrape'
    # ),



    # ############################# D3 dataset #########################################
    ############################ WITH TRANSFORMATIONS #########################################
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/augm/gen_0",
    #     data_mode="ours",
    #     key="augm_gen0_and_real",
    # ),
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/augm/gen_1",
    #     data_mode="ours",
    #     key="augm_gen_1",
    # ),
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/augm/gen_2",
    #     data_mode="ours",
    #     key="augm_gen_2",
    # ),
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/augm/gen_3",
    #     data_mode="ours",
    #     key="augm_gen_3",
    # ),
    ############################ WITHOUT TRANSFORMATIONS #########################################
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/no_augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/no_augm/gen_0",
    #     data_mode="ours",
    #     key="no_augm_gen0_and_real",
    # ),
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/no_augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/no_augm/gen_1",
    #     data_mode="ours",
    #     key="no_augm_gen_1",
    # ),
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/no_augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/no_augm/gen_2",
    #     data_mode="ours",
    #     key="no_augm_gen_2",
    # ),
    # dict(
    #     real_path="/data/lp/CoDE/d3_test/no_augm/real",
    #     fake_path="/data/lp/CoDE/d3_test/no_augm/gen_3",
    #     data_mode="ours",
    #     key="no_augm_gen_3",
    # ),
    # ############################ diffusion datasets #########################################
    # dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/dalle/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/dalle/1_fake",
    #     data_mode="wang2020",
    #     key="dalle",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/glide_50_27/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/glide_50_27/1_fake",
    #     data_mode="wang2020",
    #     key="glide_50_27",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/glide_100_10/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/glide_100_10/1_fake",
    #     data_mode="wang2020",
    #     key="glide_100_10",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/glide_100_27/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/glide_100_27/1_fake",
    #     data_mode="wang2020",
    #     key="glide_100_27",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/guided/0_real_imagenet",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/guided/1_fake",
    #     data_mode="wang2020",
    #     key="guided",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/ldm_100/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/ldm_100/1_fake",
    #     data_mode="wang2020",
    #     key="ldm_100",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/ldm_200/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/ldm_200/1_fake",
    #     data_mode="wang2020",
    #     key="ldm_200",
    # ),
    #     dict(
    #     real_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/ldm_200_cfg/0_real_laion",
    #     fake_path="/data/lp/NPR-DeepfakeDetection/UniFDData/diffusion_datasets/ldm_200_cfg/1_fake",
    #     data_mode="wang2020",
    #     key="ldm_200_cfg",
    # ),
]
