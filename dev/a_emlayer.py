from os.path import join

from . import memory

depend_extra = (memory,)

options = dict(filename='')

class nndata():
	def __init__(self, filename):
		self.filename = filename
		self.fh = open(filename, 'rt')

	def nextlayer(self):
		line = self.fh.readline().rstrip('\n').split(' ')
		self.hi, self.wi, self.ci, self.groups, self.k, self.co, self.stride = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		self.ho, self.wo, _, self.pad = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')

		def getdata():
			x = self.fh.readline()
			return tuple(map(float, x.rstrip().split(' ')))

		self.weights = getdata()
		self.biases  = getdata()
		self.inputs  = getdata()
		self.outputs = getdata()



def prepare(SOURCE_DIRECTORY):
	nn = nndata(join(SOURCE_DIRECTORY, options.filename))

	WL = 32
	xmem = memory.Memory(224*224*3, WL)
#	ymem = Memory(112*112*5, WL)

	for x in range(54):
		print()
		print(x)
		nn.nextlayer()
		if nn.k == 1:
			print(nn.wi, nn.hi, nn.ci, '->', nn.wo, nn.ho, nn.co)
			print(nn.k, nn.groups, nn.stride, nn.pad)

			memory.feat2mem(xmem, nn.inputs, width=nn.wi, height=nn.hi, channels=nn.ci)
			
