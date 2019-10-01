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

		l0 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (loepnummer-1,)))
		l = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (loepnummer,)))
		l2 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (loepnummer+1,)))

		xmem = memory.Memory(112*112*5, WL)
		ymem = memory.Memory(112*112*5, WL)

		wmem0 = memory.create_weight_mem_1x1(l0.weights, nwords=WL, channels_in=l0.ci, channels_out=l0.co)
		bias0 = convlayer.BiasNorm(l0)
		wmem1 = memory.create_weight_mem_3x3dw(l.weights, nwords=WL, channels=l.ci)
		bias1 = convlayer.BiasNorm(l)
		wmem2 = memory.create_weight_mem_1x1(l2.weights, nwords=WL, channels_in=l2.ci, channels_out=l2.co)
		bias2 = convlayer.BiasNorm(l2)

		xmem.importvec(l0.inputs, width=l0.wi, height=l0.hi, channels=l0.ci)
		xmem.readcnt = 0

		# input
		def xmemread(w, h):
			# return data for all channels at (w, h)
			v = []
			for c in range(ceil(l0.ci/WL)):
				a = w + (h * l0.wi) + (c * l0.wi * l0.hi)
				v.append(xmem.read(a))
			return v

		cache0 = cache.FuncCache(1000000, func=xmemread)
		layer0 = convlayer.Conv1x1_block(cache0.read, wmem0, l0.wi, l0.hi, l0.ci, l0.co, bias0, WL, name='l0')
		cache01 = cache.FuncCache(1000000, func=layer0.conv)
		layer = convlayer.Conv3x3dw_block(cache01.read, wmem1, l.wi, l.hi, l.ci, l.stride, bias1, WL, name='l1')
		cache12 = cache.FuncCache(1000000, func=layer.conv)
		layer2 = convlayer.Conv1x1_block(cache12.read, wmem2, l2.wi, l2.hi, l2.ci, l2.co, bias2, WL, name='l2')

		# output
		for h in range(l2.ho):
			for w in range(l2.wo):
				data = layer2.conv(w, h)
				for ix, block in enumerate(data):
					ymem.write(w + h*l2.wo + ix * l2.wo * l2.ho, block)
		out = ymem.export(width=l2.wo, height=l2.ho, channels=l2.co)

		_, _, maxerr, snr = check(out, l2.outputs3)

