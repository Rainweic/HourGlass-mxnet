# Train Stacked Hourglass model with MPII

import mxnet as mx
from mxnet import gluon

from config.config_args import args
from model.HourGlass import getHourGlass
from data_loader.mpii_data import MPIIData

# ------------gen data--------------

json_file = "./train_data/mpii_annotations.json"
imgpath = "./train_data/images/"

dataset = MPIIData(
    json_file, 
    imgpath, 
    (args.inRes, args.inRes), 
    (args.outRes, args.outRes), 
    is_train = True,
    sigma = args.sigma,
    rot_flag = args.isRot,
    scale_flag = args.isFlip,
    flip_flag = args.isScale)

dataloader = gluon.data.DataLoader(
    dataset, 
    batch_size = 4,
    shuffle = True,
    num_workers = args.nWorkers)

# ------------get model------------

if args.useGPU:
    ctx = mx.gpu()
else:
    ctx = mx.cpu()
model = getHourGlass(ctx=ctx)

# ------------gen Trainer------------

for cropimg, heatmap in dataloader:
    print(cropimg)