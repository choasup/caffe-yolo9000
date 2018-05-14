f = open('select.map.names', 'r')
f1 = open('select.names', 'w')
f2 = open('select.tree', 'r')
f3 = open('select.imagenetID', 'w')
f4 = open('select.cocoID', 'w')
f5 = open('select.map', 'w')	#select137 -> yolo9000

f1.write('#select299 yolo9000 name\n')
f3.write('#imagenetID select299 yolo9000\n')
f4.write('#coocID select299 yolo9000\n')

name = {}
imagenetID = {}
cocoID = {}
selectID = []

for line in f:
	lines = line.split(';')
	selectID.append(lines[0])
	name[lines[0]] = lines[-1][1:-2]	#select_id (s0001) -> name
	imagenetID[(lines[0])] = lines[1]	#select_id -> imagenet_id
	
	if lines[2] != '':
		cocoID[int(lines[2])] = lines[0] #coco_id (1, 90) ->  select137 (0, 293)

id_index = {}
index = 0

for line in f2:
	lines = line.split(' ')
	id_index[(lines[0])] = str(index)	#select 137 (0, 293) -> training yolo9000 (0, 293)
		
	f1.write(lines[0])	#select137
	f1.write(' ')
	f1.write(str(index))	#yolo9000
	f1.write(' ')
	f1.write(name[lines[0]])	#name
	f1.write('\n')
			
	f3.write(imagenetID[lines[0]])	#imagenetID
	f3.write(' ')
	f3.write(lines[0])	#select137
	f3.write(' ')
	f3.write(str(index)) #yolo9000
	f3.write('\n')
	index += 1

for x in sorted(cocoID):
	f4.write(str(x))	#cocoID
	f4.write(' ')
	f4.write(cocoID[x])	#select137
	f4.write(' ')
	f4.write(id_index[(cocoID[x])])	#yolo9000
	f4.write('\n')

for x in sorted(id_index):	# x: selectID -> yolo 9000
	#f5.write(x + ' ')
	f5.write(id_index[x])	# yolo 9000.
	f5.write('\n')


f1.close()
f3.close()
f4.close()
f5.close()
f.close()

selectIndex = {}
index = 0
for x in sorted(selectID):
	selectIndex[x] = index
	index += 1

fl = open('labelmap_select.prototxt', 'w')
for x in sorted(imagenetID):
	n = imagenetID[x]	#imagenetID
	l = selectIndex[x]		#select299 -> id_index(yolo 9000).
	d_name = name[x]	#display_name
	s = '\n  name: "{}"\n  label: {}\n  display_name: "{}"\n'.format(n, l, d_name)
	
	fl.write('item{')
	fl.write(s)
	fl.write('}\n')

for x in sorted(cocoID):
	n = str(x)	#cocoID
	l = selectIndex[cocoID[x]] #cocoID -> select299 id -> yolo 9000.
	d_name = name[cocoID[x]]	#display_name
	s = '\n  name: "{}"\n  label: {}\n  display_name: "{}"\n'.format(n, l, d_name)
	
	fl.write('item{')
	fl.write(s)
	fl.write('}\n')

fl.close()
