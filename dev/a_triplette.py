from collections import namedtuple
from math import log10

from extras import resolve_jobid_filename

from . import darknetlayer
from . import memory
from . import convlayer

depend_extra = (memory, convlayer, darknetlayer)

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

		# input data
		xmem.importvec(l0.inputs, width=l0.wi, height=l0.hi, channels=l0.ci)
		# first layer
		wmem = memory.create_weight_mem_1x1(l0.weights, nwords=WL, channels_in=l0.ci, channels_out=l0.co)
		bias = convlayer.BiasNorm(l0)
		convlayer.conv1x1_block(xmem, ymem, wmem, width=l0.wi, height=l0.hi, channels_in=l0.ci, channels_out=l0.co, bias=bias)
		# middle layer
		xmem, ymem = ymem, xmem
		wmem = memory.create_weight_mem_3x3dw(l1.weights, nwords=WL, channels=l1.ci)
		bias = convlayer.BiasNorm(l1)
		convlayer.conv3x3dw_block(xmem, ymem, wmem, l1.wi, l1.hi, l1.ci, bias)
		# last layer
		xmem, ymem = ymem, xmem
		wmem = memory.create_weight_mem_1x1(l2.weights, nwords=WL, channels_in=l2.ci, channels_out=l2.co)
		bias = convlayer.BiasNorm(l2)
		convlayer.conv1x1_block(xmem, ymem, wmem, width=l2.wi, height=l2.hi, channels_in=l2.ci, channels_out=l2.co, bias=bias)
		# output data
		out = ymem.export(width=l2.wo, height=l2.ho, channels=l2.co)

		check(out, l2.outputs3)

