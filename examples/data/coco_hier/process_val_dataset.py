import shutil
import os

data_root = '/home/data/liuyong/choas/datasets/MSCOCO/coco/images/val2014'
#dirs = os.listdir('/home/data/liuyong/choas/datasets/ImageNet2014/Detection/ILSVRC2014_DET_train')
dirs = os.listdir(data_root)
f = open('val_coco.txt', 'w')

for x in dirs:
    f.write(x)
    f.write('\n')

f.close()
