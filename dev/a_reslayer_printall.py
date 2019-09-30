from dataset import Dataset

datasets = ('source',)

fmt = "  {:40s} {:14,} {:14,} {:14,} {:14,} {:14,} {:14,}"


def synthesis():

	v = []
	for name in ('optimal', 'full', 'partial3'):
		ds = Dataset(datasets.source + '/' + name)
		v.append(name)
		v.append("  layer                                              mac1           mac2           mac3            cc1            cc2            cc3")
		s = [0 for x in range(6)]
		for data in ds.iterate(None, ('name', 'mac1', 'mac2', 'mac3', 'cc1', 'cc2', 'cc3')):
			v.append(fmt.format(*data))
			s = [s[ix] + data[ix+1] for ix in range(6)]
		v.append(fmt.format(*(['sum per column',] + s)))
		v.append(fmt.format('sum(mac), sum(cc)', 0, 0, sum(s[0:3]), 0, 0, sum(s[3:6])))

	for item in v:
		print(str(v))
