from collections import namedtuple
from math import log10

from extras import resolve_jobid_filename

from . import darknetlayer
from . import memory
from . import convlayer_classes as convlayer
from . import cache

depend_extra = (memory, convlayer, darknetlayer, cache)

jobids  = ('darknet',) # directory with inputs/weights/outputs, one file per layer
datasets= ('config',)  # dataset with network configuration

options = dict(cache12size=100000, cache01size=100000)



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
	columns = ('ci','sx','co','wo','layer','wi','n','mac','stride','loepnummer','hi','ho','groups','sy')
	config = namedtuple('config', columns)
	v = tuple(x for x in map(lambda x: config(*x), datasets.config.iterate(None, columns)))

	print('Bottlenet combos, not counting stride==2 ones:')
	n = 0
	triplettes = {}
	for ix in range(2, len(v)):
		d2, d1, d0 = v[ix-2:ix+1]
		if d2.sx == 1 and \
		   d1.sx == 3 and d1.stride == 1 and d1.groups == d1.ci == d1.co and\
		   d0.sx == 1:
			triplettes[n] = (d2,d1,d0)
			n += 1



#####################################################
	res = []

	for n, data in sorted(triplettes.items()):
		print('\n\n')
		print('%2d  ' %(n,) + str(data[0]))
		print('%2d  ' %(n,) + str(data[1]))
		print('%2d  ' %(n,) + str(data[2]))

		# run three layers in turn, compare output to golden
		l0 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (data[0].loepnummer,)))
		l1 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (data[1].loepnummer,)))
		l2 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (data[2].loepnummer,)))

		WL = 32
		xmem = memory.Memory(224*224*3, WL)
		ymem = memory.Memory(112*112*5, WL)

		wmem0 = memory.create_weight_mem_1x1(l0.weights, nwords=WL, channels_in=l0.ci, channels_out=l0.co)
		bias0 = convlayer.BiasNorm(l0)
		wmem1 = memory.create_weight_mem_3x3dw(l1.weights, nwords=WL, channels=l1.ci)
		bias1 = convlayer.BiasNorm(l1)
		wmem2 = memory.create_weight_mem_1x1(l2.weights, nwords=WL, channels_in=l2.ci, channels_out=l2.co)
		bias2 = convlayer.BiasNorm(l2)

		print('GurGel')



		# input data
		xmem.importvec(l0.inputs, width=l0.wi, height=l0.hi, channels=l0.ci)


		# first layer
		def xreadfun0(coord):
			w, h, c = coord
			a = w + (h * l0.wi) + (c * l0.wi * l0.hi)
			return xmem.read(a)
		layer0 = convlayer.Conv1x1_block(xreadfun0, wmem0, l0.wi, l0.hi, l0.ci, l0.co, bias0, WL)

		cache01 = cache.FuncCache(options.cache01size, stride=1, func=layer0.conv)

		layer1 = convlayer.Conv3x3dw_block(cache01.read, wmem1, l1.wi, l1.hi, l1.ci, bias1, WL)

		cache12 = cache.FuncCache(options.cache12size, stride=1, func=layer1.conv)

		layer2 = convlayer.Conv1x1_block(cache12.read, wmem2, l2.wi, l2.hi, l2.ci, l2.co, bias2, WL)

		for h in range(l2.hi):
			for w in range(l2.wi):
				data = layer2.conv(w, h)
				for ix, block in enumerate(data):
					ymem.write(w + h*l0.wi + ix * l0.wi * l0.hi, block)

		# output data
		out = ymem.export(width=l2.wo, height=l2.ho, channels=l2.co)

		_, _, maxerr, snr = check(out, l2.outputs3)

		print(cache01.m.reads, cache01.m.hits, cache01.m.miss)
		print(cache12.m.reads, cache12.m.hits, cache12.m.miss)

		res.append((n, maxerr, snr, (cache01.m.reads, cache01.m.hits, cache01.m.miss), (cache12.m.reads, cache12.m.hits, cache12.m.miss)))

	return res
