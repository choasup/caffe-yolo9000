# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
# display plots in this notebook

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import math
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

caffe.set_mode_gpu()

#model_def = './res101_animals/ResNet-101-deploy.prototxt'
#model_weights = './res101_animals/resnet-101-voc_iter_15000.caffemodel'

model_def = './../gnet_region_deploy_darknet448_voc.prototxt'
model_weights = './../models/gnet_yolo_region_darknet448_voc__iter_72500.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

mu = np.array([104, 117, 123])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          544, 544)  # image size is 227x227

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        
        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1
    
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res


def det(image, image_id):
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()

	res = output['conv_reg'][0]  # the output probability vector for the first image in the batch
	side = 17
	classes = 20
        num = 5
	pred = classes + 4 + 1
        swap = np.zeros((side * side, num, pred))
	

	#change
	index = 0
	for h in range(side):
    		for w in range(side):
        		for c in range(pred * num):
            			swap[h * side + w][c / (pred)][c % (pred)]  = res[c][h][w]


	#biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
	biases = [1.32, 1.73, 3.19, 4.01, 5.06, 8.10, 9.47, 4.84, 11.24, 10.01]
	
	boxes = list()
	for h in range(side):
    		for w in range(side):
        		for n in range(num):
            			box = list();
            			cls = list();
            			s = 0;
            			x = (w + sigmoid(swap[h * side + w][n][0])) * 1.0 / side;
            			y = (h + sigmoid(swap[h * side + w][n][1])) * 1.0 / side;
            			ww = (math.exp(swap[h * side + w][n][2]) * biases[2*n]) * 1.0 / side;
            			hh = (math.exp(swap[h * side + w][n][3])*biases[2*n+1]) * 1.0 / side;
            			obj_score = sigmoid(swap[h * side + w][n][4]);
            			for p in range(classes):
                			cls.append(swap[h * side + w][n][5 + p]);
            
            			large = max(cls);
            			for i in range(len(cls)):
                			cls[i] = math.exp(cls[i] - large);
            
            			s = sum(cls);
            			for i in range(len(cls)):
                			cls[i] = cls[i] * 1.0 / s;
                
            			box.append(x);
            			box.append(y);
            			box.append(ww);
            			box.append(hh);
            			box.append(cls.index(max(cls))+1)
            			box.append(obj_score);
            			box.append(max(cls));
				box.append(obj_score * max(cls))
            
            			if box[5] * box[6] > 0.1:
                			boxes.append(box);
	res = apply_nms(boxes, 0.45)
	label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

	w = image.shape[1]
	h = image.shape[0]

	plt.imshow(image)
        currentAxis = plt.gca()
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

	res_name = "./results/comp4_det_test_";
	for box in res:
    		name = res_name + label_name[box[4]]
		#print name
    		fid = open(name+".txt", 'a')
    		fid.write(image_id)
    		fid.write(' ')
    		fid.write(str(box[5]*box[6]))
    		fid.write(' ')
		
		xmin = (box[0]-box[2]/2.0) * w;
		xmax = (box[0]+box[2]/2.0) * w;
		ymin = (box[1]-box[3]/2.0) * h;
		ymax = (box[1]+box[3]/2.0) * h;
	        if xmin < 0:
			xmin = 0
		if xmax > w:
			xmax = w
		if ymin < 0:
			ymin = 0
		if ymax > h:
			ymax = h
		
		fid.write(str(xmin))
		fid.write(' ')
		fid.write(str(ymin))
		fid.write(' ')
		fid.write(str(xmax))
		fid.write(' ')
		fid.write(str(ymax))
		fid.write('\n')

		ids = label_name[box[4]]	
		
		#print xmin, xmax, ymin, ymax, ids
    		display_txt = '{}, {:0.2f} '.format(label_name[box[4]], box[7])
        
        	coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        	color = colors[box[4]]
        	currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        	currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    	
	plt.savefig("darknet544/" + image_id + ".png", dpi = 120)
	plt.cla()
#image = caffe.io.load_image('./images/horses.jpg')
data_root = '/home/data/liuyong/choas/datasets/PASCOL_VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
f = open('07test.txt', 'r')
index = 0
for x in f:
	index += 1
	#if index > 1:
	#	break
	ids = (x.split(' ')[0]).split('/')[-1]	
	image = caffe.io.load_image(data_root + ids)
	print index, ids[:-4]
	det(image, ids[:-4])
