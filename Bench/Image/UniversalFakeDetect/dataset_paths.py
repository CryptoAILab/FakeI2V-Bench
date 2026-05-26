# DATASET_PATHS = [


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/progan',     
#         fake_path='../FAKE_IMAGES/CNN/test/progan',
#         data_mode='wang2020',
#         key='progan'
#     ),

#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/cyclegan',   
#         fake_path='../FAKE_IMAGES/CNN/test/cyclegan',
#         data_mode='wang2020',
#         key='cyclegan'
#     ),

#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/biggan/',   # Imagenet 
#         fake_path='../FAKE_IMAGES/CNN/test/biggan/',
#         data_mode='wang2020',
#         key='biggan'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/stylegan',    
#         fake_path='../FAKE_IMAGES/CNN/test/stylegan',
#         data_mode='wang2020',
#         key='stylegan'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/gaugan',    # It is COCO 
#         fake_path='../FAKE_IMAGES/CNN/test/gaugan',
#         data_mode='wang2020',
#         key='gaugan'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/stargan',  
#         fake_path='../FAKE_IMAGES/CNN/test/stargan',
#         data_mode='wang2020',
#         key='stargan'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/deepfake',   
#         fake_path='../FAKE_IMAGES/CNN/test/deepfake',
#         data_mode='wang2020',
#         key='deepfake'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/seeingdark',   
#         fake_path='../FAKE_IMAGES/CNN/test/seeingdark',
#         data_mode='wang2020',
#         key='sitd'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/san',   
#         fake_path='../FAKE_IMAGES/CNN/test/san',
#         data_mode='wang2020',
#         key='san'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/crn',   # Images from some video games
#         fake_path='../FAKE_IMAGES/CNN/test/crn',
#         data_mode='wang2020',
#         key='crn'
#     ),


#     dict(
#         real_path='../FAKE_IMAGES/CNN/test/imle',   # Images from some video games
#         fake_path='../FAKE_IMAGES/CNN/test/imle',
#         data_mode='wang2020',
#         key='imle'
#     ),
    

#     dict(
#         real_path='./diffusion_datasets/imagenet',
#         fake_path='./diffusion_datasets/guided',
#         data_mode='wang2020',
#         key='guided'
#     ),


#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/ldm_200',
#         data_mode='wang2020',
#         key='ldm_200'
#     ),

#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/ldm_200_cfg',
#         data_mode='wang2020',
#         key='ldm_200_cfg'
#     ),

#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/ldm_100',
#         data_mode='wang2020',
#         key='ldm_100'
#      ),


#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/glide_100_27',
#         data_mode='wang2020',
#         key='glide_100_27'
#     ),


#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/glide_50_27',
#         data_mode='wang2020',
#         key='glide_50_27'
#     ),


#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/glide_100_10',
#         data_mode='wang2020',
#         key='glide_100_10'
#     ),


#     dict(
#         real_path='./diffusion_datasets/laion',
#         fake_path='./diffusion_datasets/dalle',
#         data_mode='wang2020',
#         key='dalle'
#     ),



# ]

DATASET_PATHS = [
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/biggan',
        fake_path='/data/lp/IVBridge-OriCode/datasets/biggan',
        data_mode='wang2020',
        key='biggan'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/crn',
        fake_path='/data/lp/IVBridge-OriCode/datasets/crn',
        data_mode='wang2020',
        key='crn'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/cyclegan',
        fake_path='/data/lp/IVBridge-OriCode/datasets/cyclegan',
        data_mode='wang2020',
        key='cyclegan'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/deepfake',
        fake_path='/data/lp/IVBridge-OriCode/datasets/deepfake',
        data_mode='wang2020',
        key='deepfake'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/dalle',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/dalle',
        data_mode='wang2020',
        key='dalle'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/glide_100_10',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/glide_100_10',
        data_mode='wang2020',
        key='glide_100_10'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/glide_100_27',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/glide_100_27',
        data_mode='wang2020',
        key='glide_100_27'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/glide_50_27',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/glide_50_27',
        data_mode='wang2020',
        key='glide_50_27'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/guided',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/guided',
        data_mode='wang2020',
        key='guided'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/ldm_100',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/ldm_100',
        data_mode='wang2020',
        key='ldm_100'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/ldm_200',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/ldm_200',
        data_mode='wang2020',
        key='ldm_200'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/ldm_200_cfg',
        fake_path='/data/lp/IVBridge-OriCode/datasets/diffusion_datasets/ldm_200_cfg',
        data_mode='wang2020',
        key='ldm_200_cfg'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/gaugan',
        fake_path='/data/lp/IVBridge-OriCode/datasets/gaugan',
        data_mode='wang2020',
        key='gaugan'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/imle',
        fake_path='/data/lp/IVBridge-OriCode/datasets/imle',
        data_mode='wang2020',
        key='imle'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/progan',
        fake_path='/data/lp/IVBridge-OriCode/datasets/progan',
        data_mode='wang2020',
        key='progan'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/san',
        fake_path='/data/lp/IVBridge-OriCode/datasets/san',
        data_mode='wang2020',
        key='san'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/seeingdark',
        fake_path='/data/lp/IVBridge-OriCode/datasets/seeingdark',
        data_mode='wang2020',
        key='seeingdark'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/stargan',
        fake_path='/data/lp/IVBridge-OriCode/datasets/stargan',
        data_mode='wang2020',
        key='stargan'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/stylegan',
        fake_path='/data/lp/IVBridge-OriCode/datasets/stylegan',
        data_mode='wang2020',
        key='stylegan'
    ),
    dict(
        real_path='/data/lp/IVBridge-OriCode/datasets/stylegan2',
        fake_path='/data/lp/IVBridge-OriCode/datasets/stylegan2',
        data_mode='wang2020',
        key='stypegan2'
    ),
    # dict(
    #     real_path='/data/lp/IVBridge-OriCode/datasets/whichfaceisreal',
    #     fake_path='/data/lp/IVBridge-OriCode/datasets/whichfaceisreal',
    #     data_mode='wang2020',
    #     key='whichfaceisreal'
    # ),
]
