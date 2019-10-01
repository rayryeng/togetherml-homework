## How to run: python3 -m torch.distributed.launch --nproc_per_node=4 ./week3_train.py
# Load in relevant packages
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import numpy as np
import torch

## New - Added for allowing FastAI to distribute training over multiple GPUs
# Reference: https://docs.fast.ai/distributed.html
from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

## Step #1 - Get the CAMVID data
path = untar_data(URLs.CAMVID)

## Step #2 - Set path to labels and images
path_lbl = path / 'labels'
path_img = path / 'images'

## Step #3 - Set up image filename to corresponding mask filename function
get_y_fn = lambda x: path_lbl / f'{x.stem}_P{x.suffix}'

## Step #4 - Load up the labels array that provides the corresponding label ID of a pixel to its actual label
codes = np.loadtxt(path / 'codes.txt', dtype=str)

## Step #5 - Get frame size
fnames = get_image_files(path_img)
img_f = fnames[0]
img = open_image(img_f)
src_size = np.array(img.shape[1:])

bs = 8 # Batch size of 8

## Step #6 - First train at half resolution
size = src_size // 2

# Create source and data loaders
src = (SegmentationItemList.from_folder(path_img) # Source of images?
        .split_by_fname_file('../valid.txt') # Split into training and validation by filename given in valid.txt
        .label_from_func(get_y_fn, classes=codes)) # Corresponding mask image given by original image filename and
                                                   # map the corresponding ID to each string

data = (src.transform(get_transforms(), size=size, tfm_y=True)
       .databunch(bs=bs)
       .normalize(imagenet_stats))

# Create custom accuracy for camvid
# Get index for where Void code is
void_code = np.where(codes == 'Void')[0][0]

# Now do it
def acc_camvid(inp, target):
    target = target.squeeze(1) # Take the entire image and reshape into a single column
    mask = target != void_code # Only consider the elements that are not void
    
    # The output should be a 2D matrix with each pixel
    # being in a row and each column being the probability
    # of that pixel belonging to that class
    # Look at the class with the largest probability, then
    # mask out the void pixels
    # Find the accuracy by examining the entire image and seeing
    # the proportion of labels aligning
    return (inp.argmax(dim=1)[mask] == target[mask]).float().mean()

# Now train the model
# Set regularization to 0.01
wd = 0.01
learn = unet_learner(data, models.resnet34, metrics=[acc_camvid], wd=wd)

### Make training distributed
learn = learn.to_distributed(args.local_rank)

# Find the optimal learning rate for dense layers only first
#learn.lr_find()

# Save the learning rates and losses to file
#output = np.array([learn.recorder.lrs, learn.recorder.losses]).T
#np.savetxt('stage1-lr_find.txt', output)

### Half-size - Stage 1
# Turns out LR is 3e-3
lr = 3e-3
# pct_start = 0.9 means for 90% of the iterations
# we are increasing our learning rate up to the max LR
# then we decrease for the remaining 10%
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
## Performance
# epoch     train_loss  valid_loss  acc_camvid  time
# 0         2.620851    1.567428    0.569883    00:33
# 1         1.848564    1.057415    0.744727    00:13
# 2         1.415269    0.775114    0.809854    00:13
# 3         1.161766    0.640451    0.842656    00:13
# 4         0.999325    0.614549    0.839922    00:13
# 5         0.949605    0.636080    0.844837    00:13
# 6         0.884796    0.581527    0.852120    00:14
# 7         0.815878    0.612123    0.850409    00:13
# 8         0.772048    0.493012    0.862959    00:13
# 9         0.706791    0.440848    0.874906    00:14
# Save the model
learn.save('stage-1')
learn.purge()

### Half-size - Stage 2
learn.load('stage-1')

# Unfreeze the model
learn.unfreeze()

# Specify new LRs
lrs = slice(lr/400, lr/4)

# Fit again - this time with 80% rise
learn.fit_one_cycle(10, lrs, pct_start=0.8)
# epoch     train_loss  valid_loss  acc_camvid  time
# 0         0.491588    0.422547    0.876979    00:33
# 1         0.487026    0.412312    0.878113    00:14
# 2         0.463965    0.393398    0.881522    00:13
# 3         0.454096    0.370825    0.886916    00:14
# 4         0.440402    0.358994    0.888719    00:14
# 5         0.424523    0.337217    0.902207    00:13
# 6         0.414357    0.336669    0.900211    00:13
# 7         0.413498    0.354613    0.894416    00:14
# 8         0.409670    0.369747    0.891610    00:14
# 9         0.402447    0.310375    0.909732    00:14
# Save
learn.save('stage-2')
learn.purge()

### Full-size - Stage 1
# Try again with larger images
# Free GPU memory and try again
learn.destroy()
size = src_size
bs = 4 # Half the batch size due to full size image

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=[acc_camvid], wd=wd)
learn = learn.to_distributed(args.local_rank) ## Important
learn.load('stage-2')
lr = 1e-3

# # Fit dense layers
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
learn.save('stage-1-big')
learn.purge()
# Performance
# epoch     train_loss  valid_loss  acc_camvid  time
# 0         0.419604    0.360355    0.897705    00:55
# 1         0.397169    0.350055    0.902159    00:38
# 2         0.360587    0.322498    0.908044    00:36
# 3         0.360859    0.379146    0.890260    00:37
# 4         0.349585    0.302784    0.915848    00:38
# 5         0.344647    0.318304    0.911416    00:37
# 6         0.340244    0.280113    0.923154    00:38
# 7         0.332244    0.360037    0.902354    00:37
# 8         0.317499    0.296204    0.916571    00:37
# 9         0.294801    0.255505    0.927551    00:37

### Full size - Stage 2
# Fit all layers
lrs = slice(1e-6, lr/10)

learn.load('stage-1-big')
learn.unfreeze()
learn.fit_one_cycle(10, lrs)
learn.save('stage-2-big')
learn.purge()
## Performance
# epoch     train_loss  valid_loss  acc_camvid  time
# 0         0.247682    0.255651    0.927368    00:59
# 1         0.241647    0.258040    0.926119    00:39
# 2         0.234325    0.247678    0.928538    00:38
# 3         0.232173    0.242731    0.931678    00:38
# 4         0.225396    0.254416    0.927466    00:39
# 5         0.222135    0.248926    0.929588    00:38
# 6         0.219277    0.233751    0.933411    00:37
# 7         0.214168    0.244000    0.930377    00:38
# 8         0.212540    0.249221    0.929638    00:39
# 9         0.216742    0.244496    0.931539    00:39

### Full size - Stage 3
# Try one last time
learn.load('stage-2-big')
learn.unfreeze()
learn.fit_one_cycle(10, lrs)
learn.save('stage-3-big')
learn.export('camvid-final.pkl')
learn.purge()
## Performance
# epoch     train_loss  valid_loss  acc_camvid  time
# 0         0.210978    0.242383    0.931870    00:59
# 1         0.210868    0.246082    0.931064    00:39
# 2         0.208431    0.258395    0.929573    00:39
# 3         0.206778    0.244322    0.932457    00:38
# 4         0.201983    0.253664    0.929462    00:37
# 5         0.202081    0.235387    0.933321    00:38
# 6         0.199577    0.242461    0.932987    00:38
# 7         0.196214    0.242802    0.932232    00:38
# 8         0.192381    0.250865    0.931186    00:37
# 9         0.196752    0.247533    0.932539    00:38

## Note - model was saved in /home/ubuntu/.fastai/data/camvid/images/models/stage-3-big.pth
# Final model saved in /home/ubuntu/.fastai/data/camvid/images/models/camvid-final.pkl