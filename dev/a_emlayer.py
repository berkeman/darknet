from os.path import join

from . import memory
from . import convlayer

depend_extra = (memory, convlayer)


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

def check(xv, yv, thres=1e-5):
	cnt = 0
	errs = 0
	maxerr = 0
	for ix, (x, y) in enumerate(zip(xv, yv)):
		e = abs(x - y)
		if e > thres:
			errs += 1
			maxerr = max(maxerr, e)
		cnt += 1
	print('checked', cnt)
	print('errs   ', errs)
	print('maxerr ', maxerr)


def prepare(SOURCE_DIRECTORY):
	nn = nndata(join(SOURCE_DIRECTORY, options.filename))

	WL = 32
	xmem = memory.Memory(224*224*3, WL)
	ymem = memory.Memory(112*112*5, WL)

	for x in range(54):
		print()
		print(x)
		nn.nextlayer()
		if nn.k == 1 and nn.groups == 1:
			print(nn.wi, nn.hi, nn.ci, '->', nn.wo, nn.ho, nn.co)
			print(nn.k, nn.groups, nn.stride, nn.pad)

			xmem.importvec(nn.inputs, width=nn.wi, height=nn.hi, channels=nn.ci)
			wmem = memory.create_weight_mem_1x1(nn.weights, nwords=WL, channels_in=nn.ci, channels_out=nn.co)
			convlayer.conv1x1_block(xmem, ymem, wmem, width=nn.wi, height=nn.hi, channels_in=nn.ci, channels_out=nn.co)
			out = ymem.export(width=nn.wo, height=nn.ho, channels=nn.co)
			

			check(out, nn.outputs)
