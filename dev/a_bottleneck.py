from math import log10, ceil

from accelerator.extras import resolve_jobid_filename, DotDict
from accelerator.status import status
from accelerator import blob
from . import darknetlayer
from . import memory
from . import convlayer_classes as convlayer
from . import cache

depend_extra = (memory, convlayer, darknetlayer, cache)

jobids  = (
	'darknet',      # directory with inputs/weights/outputs, one file per layer
	'bottlenecks',  # job with pickled list of bottleneck-layers
)

options = dict(
	xmemsize=224*224*3,
	cache0size=1,
	cache01size=1,
	cache12size=1,
	WL = 32,
	runonly=None,
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



def prepare(params):
	# load all bottlenecks and spread "evenly" to all slices
	x, _ = blob.load(jobid=jobids.bottlenecks)
	res = [{} for x in range(params.slices)]
	for ix, (key, val) in enumerate(sorted(x.items())):
		res[ix % params.slices][key] = val
	return res



def analysis(sliceno, prepare_res):
	WL = options.WL

	bottlenecks = prepare_res[sliceno]

	res = []

	for n, data in sorted(bottlenecks.items()):
		if options.runonly and n != options.runonly:
			print('skip', n)
			continue

		print('%2d  ' %(n,) + str(data[0]))
		print('%2d  ' %(n,) + str(data[1]))
		print('%2d  ' %(n,) + str(data[2]))

		l0 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (data[0].loepnummer,)))
		l1 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (data[1].loepnummer,)))
		l2 = darknetlayer.Layer(resolve_jobid_filename(jobids.darknet, 'data_layer_%d.txt' % (data[2].loepnummer,)))

		xmem = memory.Memory(options.xmemsize, WL)
		ymem = memory.Memory(112*112*5, WL)

		wmem0 = memory.create_weight_mem_1x1(l0.weights, nwords=WL, channels_in=l0.ci, channels_out=l0.co)
		bias0 = convlayer.BiasNorm(l0)
		wmem1 = memory.create_weight_mem_3x3dw(l1.weights, nwords=WL, channels=l1.ci)
		bias1 = convlayer.BiasNorm(l1)
		wmem2 = memory.create_weight_mem_1x1(l2.weights, nwords=WL, channels_in=l2.ci, channels_out=l2.co)
		bias2 = convlayer.BiasNorm(l2)

		# input data
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

		# bottleneck
		cache0 = cache.FuncCache(options.cache0size, func=xmemread)
		layer0 = convlayer.Conv1x1_block(cache0.read, wmem0, l0.wi, l0.hi, l0.ci, l0.co, bias0, WL, name='l0')
		cache01 = cache.FuncCache(options.cache01size, func=layer0.conv)
		layer1 = convlayer.Conv3x3dw_block(cache01.read, wmem1, l1.wi, l1.hi, l1.ci, l1.stride, bias1, WL, name='l1')
		cache12 = cache.FuncCache(options.cache12size, func=layer1.conv)
		layer2 = convlayer.Conv1x1_block(cache12.read, wmem2, l2.wi, l2.hi, l2.ci, l2.co, bias2, WL, name='l2')

		# output
		msg = "Calculating rows %%d/%d" % (l2.hi,)
		with status(msg % (0,)) as update:
			for h in range(l2.ho):
				for w in range(l2.wo):
					update((msg % (h,)) + " %d"%(w,))
					data = layer2.conv(w, h)
					for ix, block in enumerate(data):
						ymem.write(w + h*l2.wo + ix * l2.wo * l2.ho, block)
		out = ymem.export(width=l2.wo, height=l2.ho, channels=l2.co)

		_, _, maxerr, snr = check(out, l2.outputs3)

		res.append(
			DotDict(
				n=n,
				maxerr=maxerr,
				snr=snr,
				c0 = DotDict(reads=cache0.m.reads,  hits=cache0.m.hits,  miss=cache0.m.miss),
				c1 = DotDict(reads=cache01.m.reads, hits=cache01.m.hits, miss=cache01.m.miss),
				c2 = DotDict(reads=cache12.m.reads, hits=cache12.m.hits, miss=cache12.m.miss),
				c0size = options.cache0size,
				c1size = options.cache01size,
				c2size = options.cache12size,
				xsize = options.xmemsize,
				xrcnt = xmem.readcnt,
				l0stat = layer0.status(),
				l1stat = layer1.status(),
				l2stat = layer2.status(),
			)
		)
	return res



def synthesis(analysis_res):
	res = []
	for item in analysis_res:
		res.extend(item)
	return res
