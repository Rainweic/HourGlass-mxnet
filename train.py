# Train Stacked Hourglass model with MPII

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxboard import SummaryWriter
from progressbar import progressbar

from config.config_args import args
from model.HourGlass import getHourGlass
from data_loader.mpii_data import MPIIData

# 使用mxboard 让训练可视化
sm = SummaryWriter(logdir="logs", flush_secs=20)

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

# ------------train------------
def train():

    loss_fc = gluon.loss.L2Loss()
    trainer = gluon.Trainer(
        net.collect_params(),
        'sgd',
        {'learning_rate': args.lr, 'wd': 0.0005, 'momentum': 0.9}
    )

    print("Training is started...")
    for epoch in progressbar(range(args.epochs)):
        for batch_times, (in_data, hm_label) in enumerate(progressbar(train_dataloader)):
            # 数据转移至GPU
            in_data = in_data.as_in_context(ctx)
            hm_label = hm_label.as_in_context(ctx)
            loss = mx.nd.zeros(shape=(8,), ctx=ctx)
            with autograd.record():
                # 前向传播
                out = net(in_data)
                # 计算Loss
                for i in range(len(out)):
                    loss = loss + loss_fc(out[i], hm_label)
            loss.backward()
            trainer.step(args.batchSize)
            loss_mean = loss.mean().asscalar()
            if batch_times % 100 == 0: 
                print("Epoch number {} [Batch Times {}] Current loss {}".format(epoch, \
                     batch_times, loss_mean))
                sm.add_scalar("one_epoch_train_lossMean", loss_mean, global_step=epoch)
                
        if epoch == 0:
            sm.add_graph(net)

    # 保存权重
    net.export("Stacked Hourglass")

train()