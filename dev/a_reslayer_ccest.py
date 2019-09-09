from collections import namedtuple, Counter

datasets = ('source',)

columns = ('name', 'wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho', 'mac1', 'mac2', 'mac3', 'macs', 'inputfeatsize', 'outputfeatsize', 'twolinecache', 'dwcache9')

ResLayer = namedtuple('ResLayer', columns)

def synthesis():
	cc1 = 0
	cc2 = 0
	cc3 = 0
	ccs = 0

	print("%-40s %13s %14s %14s %14s %14s %14s" % ('name', '2cache', 'dw9cache', 'mac1', 'mac2', 'mac3', 'macs'))
	for x in map(lambda x: ResLayer(*x), datasets.source.iterate(None, columns)):
		print("%-40s" % (x.name,) + "{:14,} {:14,} {:14,} {:14,} {:14,} {:14,}".format(x.twolinecache, x.dwcache9, x.mac1, x.mac2, x.mac3, x.macs))
		cc1 += x.mac1
		cc2 += x.mac2
		cc3 += x.mac3
		ccs += x.macs
	print(" "*70 + "{:14,} {:14,} {:14,} {:14,}".format(cc1, cc2, cc3, ccs))
