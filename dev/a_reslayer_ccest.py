from collections import namedtuple
from math import ceil

from dataset import DatasetWriter

datasets = ('source',)
options = dict(N = 16)


columns = ('name', 'wi', 'hi', 'ci', 'cm', 'co', 'wo', 'ho', 'inputfeatsize', 'outputfeatsize', 'twolinecache', 'dwcache9')

ResLayer = namedtuple('ResLayer', columns)

ncolums = ('mac1', 'mac2', 'mac3', 'cc1', 'cc2', 'cc3')

def synthesis():

	N = options.N

	def setupds(name):
		dw = DatasetWriter(name=name, parent=datasets.source)
		for key in ncolums:
			dw.add(key, 'number')
		dw.set_slice(0)
		return dw

	dw = setupds('optimal')
	for x in map(lambda x: ResLayer(*x), datasets.source.iterate(None, columns)):
		mac1 = x.wi*x.hi * x.ci * x.cm
		mac2 = x.wo*x.ho *    9 * x.cm
		mac3 = x.wo*x.ho * x.cm * x.co
		cc1 = x.wi*x.hi * ceil(x.ci/N) * ceil(x.cm/N)
		cc2 = x.wo*x.ho * 9*ceil(x.cm/N)
		cc3 = x.wo*x.ho * ceil(x.cm/N) * ceil(x.co/N)
		dw.write(mac1, mac2, mac3, cc1, cc2, cc3)
	dw.finish()

	dw = setupds('full')
	for x in map(lambda x: ResLayer(*x), datasets.source.iterate(None, columns)):
		mac1 = 9 * x.wi*x.hi * x.ci * x.cm
		mac2 = x.wo*x.ho *    9 * x.cm
		mac3 = x.wo*x.ho * x.cm * x.co
		cc1 = 9 * x.wi*x.hi * ceil(x.ci/N) * ceil(x.cm/N)
		cc2 = x.wo*x.ho * 9*ceil(x.cm/N)
		cc3 = x.wo*x.ho * ceil(x.cm/N) * ceil(x.co/N)
		dw.write(mac1, mac2, mac3, cc1, cc2, cc3)
	dw.finish()

	dw = setupds('partial3')
	for x in map(lambda x: ResLayer(*x), datasets.source.iterate(None, columns)):
		mac1 = 3 * x.wi*x.hi * x.ci * x.cm
		mac2 = x.wo*x.ho *    9 * x.cm
		mac3 = x.wo*x.ho * x.cm * x.co
		cc1 = 3 * x.wi*x.hi * ceil(x.ci/N) * ceil(x.cm/N)
		cc2 = x.wo*x.ho * 9*ceil(x.cm/N)
		cc3 = x.wo*x.ho * ceil(x.cm/N) * ceil(x.co/N)
		dw.write(mac1, mac2, mac3, cc1, cc2, cc3)
	dw.finish()
