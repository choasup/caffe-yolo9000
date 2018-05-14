f = open('12trainval.txt', 'r')

S12 = []
for lines in f:
	S12.append(lines[:-1])

f1 = open('07trainval.txt', 'r')
f2 = open('0712trainval.txt', 'w')

for lines in f1:
	f2.write(lines)

for p in S12:
	l = 'VOC2012/JPEGImages/' + p + '.jpg ' + 'VOC2012/Annotations/' + p + '.xml\n'
	f2.write(l)
