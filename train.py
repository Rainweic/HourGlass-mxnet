# Train Stacked Hourglass model with MPII

import mxnet as mx
from mxnet import gluon

from config.config_args import args
from model.HourGlass import getHourGlass
from data_loader.mpii_data import MPIIData

# ------------gen data--------------

json_file = "./train_data/mpii_annotations.json"
imgpath = "./train_data/images/"

train_dataset = MPIIData(
    json_file, 
    imgpath, 
    (args.inRes, args.inRes), 
    (args.outRes, args.outRes), 
    is_train = True,
    sigma = args.sigma,
    rot_flag = args.isRot,
    scale_flag = args.isFlip,
    flip_flag = args.isScale)

train_dataloader = gluon.data.DataLoader(
    train_dataset, 
    batch_size = args.batchSize,
    shuffle = True,
    num_workers = args.nWorkers)

val_dataset = MPIIData(
    json_file, 
    imgpath, 
    (args.inRes, args.inRes), 
    (args.outRes, args.outRes), 
    is_train = False
)

val_dataloader = gluon.data.DataLoader(
    val_dataset,
    batch_size = args.batchSize,
    shuffle = False,
    num_workers = args.nWorkers
)

# ------------get model------------

if args.useGPU:
    ctx = mx.gpu()
else:
    ctx = mx.cpu()
net = getHourGlass(ctx)

# ------------gen Trainer------------

trainer = gluon.Trainer(
    net.collect_params(),
    'sgd',
    {'learning_rate': args.lr, 'wd': 0.0005, 'momentum': 0.9}
)

# ------------Loss------------

loss = gluon.loss.L2Loss()

# ------------train------------

for cropimg, heatmap in train_dataloader:
    print(cropimg)