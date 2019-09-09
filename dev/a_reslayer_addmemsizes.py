from dataset import DatasetWriter

datasets=('reslayers',)

options = dict(N = 16)

def synthesis():

	dw = DatasetWriter(parent=datasets.reslayers)
	dw.add('name', 'unicode')
	dw.add('mac1', 'number')
	dw.add('mac2', 'number')
	dw.add('mac3', 'number')
	dw.add('macs', 'number')
	dw.add('inputfeatsize', 'number')
	dw.add('outputfeatsize', 'number')
	dw.add('twolinecache', 'number')
	dw.add('dwcache9', 'number')
	dw.set_slice(0)

	for wi, hi, ci, cm, co, wo, ho in datasets.reslayers.iterate(None, ('wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho',)):
		name = "%dx%d-%d -> %dx%d-%d -> %dx%d-%d" % (wi, hi, ci,   wi, hi, cm,   wo, ho, co,)
		inputfeature  = wi * hi * ci
		outputfeature = wo * ho * co
		twolinechache = wi * 2 * ci
		dwcache9      = 9 * cm
		print(inputfeature, twolinechache, dwcache9)

		mac1 = wi*hi * ci * cm
		mac2 = wo*ho *  9 * cm
		mac3 = wo*ho * cm * co
		dw. write(name, mac1, mac2, mac3, mac1+mac2+mac3, inputfeature, outputfeature, twolinechache, dwcache9)
