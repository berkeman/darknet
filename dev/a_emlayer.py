from math import log10

from dataset import DatasetWriter

from extras import resolve_jobid_filename

from . import darknetlayer
from . import memory
from . import convlayer

depend_extra = (memory, convlayer, darknetlayer)


options = dict(layers=54)
jobids  = ('darknet',) # directory with inputs/weights/outputs, one file per layer
datasets= ('config',)  # dataset with network configuration



def check(xv, yv, thres=1e-5):
	cnt = 0
	errs = 0
	maxerr = 0
	perr = 0
	ptot = 0
	for ix, (x, y) in enumerate(zip(xv, yv)):
		e = abs(x - y)
		perr += e*e
		ptot += x*x
		if e > thres:
			errs += 1
		maxerr = max(maxerr, e)
		cnt += 1
	print('checked %9d  errs %9d  maxerr [42m%0.12f[0m  snr(dB) %4.2f' % (
		cnt, errs, maxerr, -10*log10(perr/ptot)))
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
	out = []

	for loepnummer in datasets.config.iterate(None, 'loepnummer'):
		if loepnummer >= options.layers:
			break
		print()
		print(loepnummer)
		nn = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (loepnummer,)))
		print('BN=%d, activation=%s' % (nn.bn, nn.activation))
		maxerr = None # scope
		if nn.k == 1 and nn.groups == 1:
			xmem.importvec(nn.inputs, width=nn.wi, height=nn.hi, channels=nn.ci)
			wmem = memory.create_weight_mem_1x1(nn.weights, nwords=WL, channels_in=nn.ci, channels_out=nn.co)
			bias = convlayer.BiasNorm(nn)
			convlayer.conv1x1_block(xmem, ymem, wmem, width=nn.wi, height=nn.hi, channels_in=nn.ci, channels_out=nn.co, bias=bias)
			out = ymem.export(width=nn.wo, height=nn.ho, channels=nn.co)
			_, _, maxerr = check(out, nn.outputs3)
			e.append(maxerr)
		elif nn.k == 3 and nn.groups == nn.ci == nn.co and nn.stride == 1:
			xmem.importvec(nn.inputs, width=nn.wi, height=nn.hi, channels=nn.ci)
			wmem = memory.create_weight_mem_3x3dw(nn.weights, nwords=WL, channels=nn.ci)
			bias = convlayer.BiasNorm(nn)
			convlayer.conv3x3dw_block(xmem, ymem, wmem, nn.wi, nn.hi, nn.ci, bias)
			out = ymem.export(width=nn.wo, height=nn.ho, channels=nn.co)
			_, _, maxerr = check(out, nn.outputs3)

			e.append(maxerr)

		print('READS', xmem.readcnt)
		dw.write(loepnummer, maxerr, xmem.readhistory)
		xmem.readhistory = []
	return (e, xmem.readcnt, convlayer.bdp.status())
