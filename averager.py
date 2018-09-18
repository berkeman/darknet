import sys

class filt:
	def __init__(self):
		self.mem = [0,0,0]
	def update(self, x):
		self.mem.append(x)
		self.mem.pop(0)
	def out(self):
		return max(self.mem)


filt6 = filt()
filt7 = filt()
filt8 = filt()

for line in sys.stdin:
	if line.startswith('#'):
		continue
	line = [x for x in line.rstrip('\n').split(' ') if x]
	filt6.update(int(line[6]))
	filt7.update(int(line[7]))
	filt8.update(int(line[8]))
	print('\t'.join(line), filt6.out(), filt7.out(), filt8.out())
