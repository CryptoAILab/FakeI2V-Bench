from util import mkdir

# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# # root to the testsets

# dataroot = '/data2/lp/IVBridge_datasets/crop/test/'

# vals = ['CDFV2', 'DFD', 'GenVideo', 'GVB']
# multiclass = [0, 0, 0, 0]

dataroot = '/data2/lp/IVBridge_datasets/crop/test/'

vals = ['split']
multiclass = [0]

# model
model_path = 'checkpoints/experiment_name/model_epoch_best.pth'
