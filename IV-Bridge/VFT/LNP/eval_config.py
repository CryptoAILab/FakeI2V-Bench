from util import mkdir

# directory to store the results
results_dir = 'results'
mkdir(results_dir)

# # root to the testsets

dataroot = '/data2/lp/IVBridge_datasets/LNP/val'


vals = ['split']

multiclass = [0]

# model
model_path = 'checkpoints/experiment_name/model_epoch_50.pth'
