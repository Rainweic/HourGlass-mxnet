import argparse

parser = argparse.ArgumentParser()

# ---------------------Model options-----------------------

parser.add_argument("--nFeats", type=int, default=256, \
    help="Number of features in the hourglass")
parser.add_argument("--nStack", type=int, default=8, \
    help="Number of hourglasses to stack")
parser.add_argument("--nModules", type=int, default=1, \
    help="Number of residual modules at each location in the hourglas")

# ---------------------Datasets options---------------------

parser.add_argument("--nJoints", type=int, default=16, \
    help="Number of dataset joints")  
parser.add_argument("--inRes", type=int, default=256, \
    help="Input image resolution")  
parser.add_argument("--outRes", type=int, default=64, \
    help="Output heatmap resolution")

# ---------------------Trainning options----------------------

parser.add_argument("--epochs", type=int, default=240)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.00025)
parser.add_argument("--sigma", type=int, default=1)
parser.add_argument("--isRot", type=bool, default=True, \
    help="Rote the train image or not")
parser.add_argument("--isFlip", type=bool, default=True, \
    help="Flip the train image or not")
parser.add_argument("--isScale", type=bool, default=True, \
    help="Scale the train image or not")

# ---------------------Running options----------------------

parser.add_argument("--useGPU", type=bool, default=False, \
    help="Use gpu or not(include train & run demo)")
parser.add_argument("--nWorkers", type=int, default=0, \
    help="The number of multiprocessing workers to use for data preprocessing.")


args = parser.parse_args()