# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
import matplotlib.patches as patches
import time
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
#caffe_root = '../../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root + 'python')

import caffe
import math
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
from numba import jit
from numpy import arange

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

def show_boxes(boxes, dh, dw, p):
    #edit 
    f = open('./../voc.name', 'r')
    label_name = {}
    index = 0
    for lines in f:
        label_name[index] = lines[:-1]
        index += 1

    
    h = image.shape[0]
    w = image.shape[1]

    #plt.imshow(image)
    #currentAxis = plt.gca()
    #colors = plt.cm.hsv(np.linspace(0, 1, 9418)).tolist()
    
    for box in boxes:
        
        x_min = int(round((box[0]-box[2]/2.0) * p)) - dw
        x_max = int(round((box[0]+box[2]/2.0) * p)) - dw
        y_min = int(round((box[1]-box[3]/2.0) * p)) - dh
        y_max = int(round((box[1]+box[3]/2.0) * p)) - dh
        
        if x_min < 0:
            x_min = 0
        if x_max > w:
            x_max = w
        if y_min < 0:
            y_min = 0
        if y_max > h:
            y_max = h
        
        display_txtes = ""
        for t in box[4]:
            display_txtes += '{}, {:0.2f} '.format(label_name[t[0]], t[1])
            display_txt = '{}, {:0.2f} '.format(label_name[t[0]], t[1])
	print display_txtes

        #coords = (x_min, y_min), x_max-x_min+1, y_max-y_min+1
        #color = colors[box[4][-1][0]]
        #currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        #currentAxis.text(x_min, y_min, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    #plt.savefig("prediction.png", dpi = 120)
    #plt.show()

def ParseTree(tree_file):
    f = open(tree_file, "r")
    parents_ = []
    child_ = []
    name_ = [] #node name
    
    groups_ = 0
    group_offset_ = [] 
    group_size_ = []
    group_ = [] #node belong to which group
    
    group_size = 0
    n = 0
    last_parent = -1
    
    for lines in f:
        line = lines.split(" ")
        name_id = line[0]
        parent = int(line[1])
        parents_.append(parent)
        child_.append(-1)
        name_.append(name_id)
        
        if parent != last_parent:
            groups_ += 1
            group_offset_.append(n - group_size)
            group_size_.append(group_size)
            group_size = 0
            last_parent = parent
        
        group_.append(groups_)
        if parent >= 0:
            child_[parent] = groups_
        
        n += 1
        group_size += 1
        
    groups_ += 1
    
    group_offset_.append(n - group_size)
    group_size_.append(group_size)
    return group_, groups_, group_size_, group_offset_, parents_, child_

start = time.time()
tree_file = "./../voc.tree"
group_, groups_, group_size_, group_offset_, parents_, child_ = ParseTree(tree_file)

caffe.set_mode_gpu()

model_def = 'gnet_region_deploy_googlenet.prototxt'
model_weights = './../models/gnet_yolo_region_googlenet_iter_5000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.array([0, 0, 0])

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

#edit by prototxt.
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          416, 416)  # image size is 227x227

@jit
def fill_image(image, dh, dw, size_p):
    new_image = np.array([0.5] * size_p * size_p * 3)
    new_image = np.reshape(new_image, [size_p, size_p, 3])

    for i in range(h):
        for j in range(w):
            for c in range(3):
                new_image[i + dh][j + dw][c] = image[i][j][c]
    return new_image    

start = time.time()
# edit side
side = 13

image = caffe.io.load_image('images/horses.jpg')
h = image.shape[0]
w = image.shape[1]
size_p = max(w, h)

dh = (int)(size_p - h) / 2
dw = (int)(size_p - w) / 2
new_image = fill_image(image, dh, dw, size_p)

transformed_image = transformer.preprocess('data', new_image)
net.blobs['data'].data[...] = transformed_image

duration = time.time() - start
print 'image prepare: {:.3f}s'.format(duration)

start = time.time()
output = net.forward()

duration = time.time() - start
print 'net forward: {:.3f}s'.format(duration)

# edit according to deploy prototxt.
res = output['conv_reg'][0]


@jit
def swap_data(res):
    # edit accoding to last layer numoutput.
    cls = 23
    n = 5	
    swap = np.zeros((side * side, n, cls + 5))  #side*side, n, cls + 5
    
    for h in range(side):
        for w in range(side):
            for c in range(n * (cls + 5)): #n * (cls + 5)
                swap[h * side + w][c / (cls + 5)][c % (cls + 5)]  = res[c][h][w]
    return swap

start = time.time()
swap = swap_data(res)
duration = time.time() - start
print 'copy data: {:.3f}s'.format(duration)

#edit by network region loss layer.
biases = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

start = time.time()

@jit
def get_max(data, h, w, n, group_size, offset):
    max_c = data[h * side + w][n][5 + offset]

    for p in range(group_size):
	if max_c < data[h * side + w][n][5 + offset + p]:
    	    max_c = data[h * side + w][n][5 + offset + p]
    return max_c

@jit
def sub_max(data, h, w, n, group_size, offset, large):
    s = 0
    for p in range(group_size):
	v = math.exp(data[h * side + w][n][5 + p + offset] - large);
	s += v
	data[h * side + w][n][5 + p + offset] = v
    return s

@jit
def div_sum(data, h, w, n, group_size, offset, sum_c):
    for p in range(group_size):
	data[h * side + w][n][5 + p + offset] = data[h * side + w][n][5 + p + offset] * 1.0 / sum_c

def active_point(data, h, w, n):
    for i in range(groups_):
    	offset = group_offset_[i]
        group_size = group_size_[i]

	large = get_max(data, h, w, n, group_size, offset)	
        sum_c = sub_max(data, h, w, n, group_size, offset, large)
	div_sum(data, h, w, n, group_size, offset, sum_c)

#def update_hier(data, h, w, n):
#    #predictions P(x) =  P(x|pa) * P(pa)
#    for i in range(9418):
#        parent = parents_[i]
#        if parent >= 0:
#            data[h * 17 + w][n][5 + i] = data[h * 17 + w][n][5 + i] * data[h * 17 + w][n][5 + parent]

def active_array(data):
    for h in range(side):
        for w in range(side):
            for n in range(3):
		active_point(data, h, w, n)
		#update_hier(data, h, w, n)
    return data

#@jit
#def active_array(data):
#    for h in range(17):
#        for w in range(17):
#            for n in range(3):
#		for i in range(groups_):
#        	    offset = group_offset_[i]
#        	    group_size = group_size_[i]
#
#            	    max_c = data[h * 17 + w][n][5 + offset]
#
#    		    for p in range(group_size):
#        		if max_c < data[h * 17 + w][n][5 + offset + p]:
#            		    max_c = data[h * 17 + w][n][5 + offset + p]
#   		    
#		    sum_c = 0
#    		    for p in range(group_size):
#        		v = math.exp(data[h * 17 + w][n][5 + p + offset] - max_c);
#        		sum_c += v
#        		data[h * 17 + w][n][5 + p + offset] = v
#
#		    for p in range(group_size):
#        		data[h * 17 + w][n][5 + p + offset] = data[h * 17 + w][n][5 + p + offset] * 1.0 / sum_c
#
#    return data

swap = active_array(swap)
duration = time.time() - start
print 'active data: {:.3f}s'.format(duration)

def get_boxes(swap, biases):
    thresh = 0.1
    boxes = []
    for h in range(side):
	for w in range(side):
	    for n in range(3):
		box = []
		
		x = (w + sigmoid(swap[h * side + w][n][0])) * 1.0 / side; #center x
                y = (h + sigmoid(swap[h * side + w][n][1])) * 1.0 / side; #center y
                ww = (math.exp(swap[h * side + w][n][2])*biases[2*n]) * 1.0  / side; #w
                hh = (math.exp(swap[h * side + w][n][3])*biases[2*n+1]) * 1.0 / side; #h
                obj_score = sigmoid(swap[h * side + w][n][4]);
       
                #top predictions
                pre = []
                pres = []
                group = 0
                p = obj_score 
		 
                while (1):
                    max_p = 0
                    max_i = 0
                
                    for i in range(group_size_[group]):
                        index = i + group_offset_[group]
                        val = swap[h * side + w][n][5 + index]
                        if (val > max_p):
                            max_p = val
                            max_i = index
                        
                    if (p * max_p > thresh):
                        pres.append([max_i, p])
                    	pre.append(max_i)
                   	p = p * max_p
                    	group = child_[max_i]
                    	if group < 0:
                            break
                    else:
                        break

           	if pre == []:
                    continue
                
                if obj_score < 0.5:
                    continue
                
                score = swap[h * side + w][n][5 + pre[-1]]
            
                if pre != []:
                    box.append(x); #0
                    box.append(y); #1
                    box.append(ww); #2
                    box.append(hh); #3
                    box.append(pres) #4
                    box.append(obj_score); #5
                    box.append(score) #6
                    box.append(obj_score * score) #7
                    boxes.append(box)

    return boxes

start = time.time()
boxes = get_boxes(swap, biases)
ress = apply_nms(boxes, 0.3)

duration = time.time() - start
print 'im_detect: {:.3f}s'.format(duration)

show_boxes(ress, dh, dw, size_p)

