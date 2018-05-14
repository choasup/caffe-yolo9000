import shutil
import os

data_root = '/home/data/liuyong/choas/datasets/ImageNet2014/Detection/ILSVRC2014_DET_train'
#dirs = os.listdir('/home/data/liuyong/choas/datasets/ImageNet2014/Detection/ILSVRC2014_DET_train')
dirs = os.listdir(data_root)
f = open('select.imagenetID', 'r')
f1 = open('train_coco_imagenet_det.txt', 'w')
f2 = open('train_coco.txt', 'r')

for line in f2:
        lines = line.split(' ')
        s1 = 'MSCOCO/coco/' + lines[0]
        s2 = 'MSCOCO/coco/' + lines[1]

        sall = s2.split('/')
        s3 = sall[0] + '/' + sall[1] + '/Annotations_xml/' + sall[3] + '/' + sall[4][:-5] + 'xml\n'
        f1.write(s1)
        f1.write(' ')
        f1.write(s3)

index = 0
for lines in f:	
	#index += 1
	#if index == 1:
	#	continue
	
	line = lines.split(' ')
	
	if line[0] in dirs:
		s1 = 'ImageNet2014/Detection/ILSVRC2014_DET_train/' + line[0]
		s2 = 'ImageNet2014/Detection/ILSVRC2014_DET_bbox_train/' + line[0]
		dirn = os.listdir(data_root + '/' + line[0])
				
		for x in dirn:
			f1.write(s1 + '/' + x)
			f1.write(' ')
			f1.write(s2 + '/' + x[:-4] + 'xml')
			f1.write('\n')

f1.close()
