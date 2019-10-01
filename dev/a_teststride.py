from math import ceil, log10

from extras import resolve_jobid_filename

from . import darknetlayer
from . import memory
from . import convlayer_classes as convlayer
from . import cache

depend_extra = (cache, memory, convlayer, darknetlayer,)


jobids  = (
	'darknet',      # directory with inputs/weights/outputs, one file per layer
)



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
	snr = -10*log10(perr/ptot)
	print('checked %9d  errs %9d  maxerr [42m%0.12f[0m  snr(dB) %4.2f' % (
		cnt, errs, maxerr, snr))
	return cnt, errs, maxerr, snr



def synthesis():

	WL = 32

	for loepnummer in (5,8,12,15,19,23,26,30,34,38,41,45,49,52,56,60):

		l = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (loepnummer,)))

		xmem = memory.Memory(112*112*5, WL)
		ymem = memory.Memory(112*112*5, WL)

		wmem1 = memory.create_weight_mem_3x3dw(l.weights, nwords=WL, channels=l.ci)
		bias1 = convlayer.BiasNorm(l)

		xmem.importvec(l.inputs, width=l.wi, height=l.hi, channels=l.ci)
		xmem.readcnt = 0

		# input
		def xmemread(w, h):
			# return data for all channels at (w, h)
			v = []
			for c in range(ceil(l.ci/WL)):
				a = w + (h * l.wi) + (c * l.wi * l.hi)
				v.append(xmem.read(a))
			return v

		cache0 = cache.FuncCache(1000000, func=xmemread)
		layer = convlayer.Conv3x3dw_block(cache0.read, wmem1, l.wi, l.hi, l.ci, l.stride, bias1, WL, name='3x3dw')

		# output
		for h in range(l.ho):
			for w in range(l.wo):
				data = layer.conv(w, h)
				for ix, block in enumerate(data):
					ymem.write(w + h*l.wo + ix * l.wo * l.ho, block)
		out = ymem.export(width=l.wo, height=l.ho, channels=l.co)

		_, _, maxerr, snr = check(out, l.outputs3)

