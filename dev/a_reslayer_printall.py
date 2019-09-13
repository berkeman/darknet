from dataset import Dataset

datasets = ('source',)

fmt = "  {:40s} {:14,} {:14,} {:14,} {:14,} {:14,} {:14,}"


def synthesis():

	for name in ('optimal', 'full', 'partial3'):
		ds = Dataset(datasets.source + '/' + name)
		print(name)
		print("  layer                                              mac1           mac2           mac3            cc1            cc2            cc3")
		s = [0 for x in range(6)]
		for data in ds.iterate(None, ('name', 'mac1', 'mac2', 'mac3', 'cc1', 'cc2', 'cc3')):
			print(fmt.format(*data))
			s = [s[ix] + data[ix+1] for ix in range(6)]
		print(fmt.format(*(['sum per column',] + s)))
		print(fmt.format('sum(mac), sum(cc)', 0, 0, sum(s[0:3]), 0, 0, sum(s[3:6])))
