from dataset import DatasetWriter

from extras import resolve_jobid_filename

from . import memory
from . import convlayer

depend_extra = (memory, convlayer)


options = dict(layers=54)
jobids  = ('darknet',) # directory with inputs/weights/outputs, one file per layer
datasets= ('config',)  # dataset with network configuration


class Layer():
	def __init__(self, filename):
		self.filename = filename
		self.fh = open(filename, 'rt')
		self.nextlayer()
	def nextlayer(self):
		line = self.fh.readline().rstrip('\n').split(' ')
		self.hi, self.wi, self.ci, self.groups, self.k, self.co, self.stride = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		self.ho, self.wo, _, self.pad = map(int, line[1:])
		line = self.fh.readline().rstrip('\n').split(' ')
		_, _, self.bn = map(int, line[1:])
		def getdata():
			x = self.fh.readline()
			return tuple(map(float, x.rstrip().split(' ')))
		self.weights = getdata()
		self.biases  = getdata()
		self.inputs  = getdata()
		self.rolling_mean = getdata()
		self.rolling_var = getdata()
		self.scales = getdata()
		self.outputs = getdata()
		self.outputs2 = getdata()


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
	return cnt, errs, maxerr


def synthesis(SOURCE_DIRECTORY):

	dw = DatasetWriter()
	dw.add('lopenummer', 'number')
	dw.add('maxerror', 'float64')
	dw.add('readhistx', 'json')
	dw.set_slice(0)

	WL = 32
	xmem = memory.Memory(224*224*3, WL)
	ymem = memory.Memory(112*112*5, WL)

	e = []

	for loepnummer in datasets.config.iterate(None, 'loepnummer'):
		if loepnummer >= options.layers:
			break
		print(loepnummer)
		nn = Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (loepnummer,)))
		print('BN:', nn.bn)
		maxerr = None # scope
		if nn.k == 1 and nn.groups == 1:
			print('1x1')
			xmem.importvec(nn.inputs, width=nn.wi, height=nn.hi, channels=nn.ci)
			wmem = memory.create_weight_mem_1x1(nn.weights, nwords=WL, channels_in=nn.ci, channels_out=nn.co)
			bias = convlayer.BiasNorm(nn)
			convlayer.conv1x1_block(xmem, ymem, wmem, width=nn.wi, height=nn.hi, channels_in=nn.ci, channels_out=nn.co, bias=bias)
			out = ymem.export(width=nn.wo, height=nn.ho, channels=nn.co)
			_, _, maxerr = check(out, nn.outputs2)
			e.append(maxerr)
		elif nn.k == 3 and nn.groups == nn.ci == nn.co and nn.stride == 1:
			print('3x3dw')
			xmem.importvec(nn.inputs, width=nn.wi, height=nn.hi, channels=nn.ci)
			wmem = memory.create_weight_mem_3x3dw(nn.weights, nwords=WL, channels=nn.ci)
			convlayer.conv3x3dw_block(xmem, ymem, wmem, nn.wi, nn.hi, nn.ci, nn.outputs)
			out = ymem.export(width=nn.wo, height=nn.ho, channels=nn.co)
			_, _, maxerr = check(out, nn.outputs)
			e.append(maxerr)

		print('READS', xmem.readcnt)
		dw.write(loepnummer, maxerr, xmem.readhistory)
		xmem.readhistory = []
	return (e, xmem.readcnt, convlayer.bdp.status())
