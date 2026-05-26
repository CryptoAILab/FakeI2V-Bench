from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = '/data/lp/IVBridge-OriCode/datasets/'

# list of synthesis algorithms
vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 
        'diffusion_datasets/dalle', 'diffusion_datasets/glide_50_27',
          'diffusion_datasets/glide_100_10', 'diffusion_datasets/glide_100_27',
            'diffusion_datasets/guided', 'diffusion_datasets/ldm_100',
        'diffusion_datasets/ldm_200', 'diffusion_datasets/ldm_200_cfg']

# indicates if corresponding testset has multiple classes
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# model
model_path = 'weights/blur_jpg_prob0.5.pth'
