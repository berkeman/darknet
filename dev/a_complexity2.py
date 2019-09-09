from math import ceil

datasets=('reslayers',)

options = dict(N = 16)

def synthesis():

	m = 0
	c = 0

	for wi, hi, ci, cm, co, wo, ho in datasets.reslayers.iterate(None, ('wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho',)):
		print()
		print("%dx%d-%d -> %dx%d-%d -> %dx%d-%d" % (wi, hi, ci,   wi, hi, cm,   wo, ho, co,))
		inputfeature  = wi * hi * ci
		twolinechache = wi * 2 * ci
		dwcache9      = 9 * cm
		print(inputfeature, twolinechache, dwcache9)

		mac1 = wi*hi * ci * cm
		mac2 = wo*ho *  9 * cm
		mac3 = wo*ho * cm * co
		m += mac1 + mac2 + mac3



		cc1 = wi*hi * ceil(ci/options.N) * ceil(cm/options.N)
		cc2 = wo*ho * 9*ceil(cm/options.N)
		cc3 = wo*ho * ceil(cm/options.N) * ceil(co/options.N)
		c+= cc1+cc2+cc3

		print(mac1, mac2, mac3, mac1+mac2+mac3, '    ', cc1, cc2, cc3, cc1+cc2+cc3)


	print(m, c)
