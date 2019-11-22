from mxnet import gluon

from data_loader.mpii_data import MPIIData

json_file = "./train_data/mpii_annotations.json"
imgpath = "./train_data/images/"


dataset = MPIIData(json_file, imgpath, (512, 512), (512, 512), True)
dataloader = gluon.data.DataLoader(dataset, batch_size=4)

for cropimg, heatmap in dataloader:
    print(type(cropimg))
    print(type(heatmap))
    print(cropimg.shape)
    print(heatmap.shape)
    print("------------------")