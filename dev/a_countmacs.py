from accelerator import blob

jobids  = (
	'bottlenecks',  # job with pickled list of bottleneck-layers
)

datasets = ('config',)

def synthesis():
	bottlenecks, unused_layers = blob.load(jobid=jobids.bottlenecks)

	botmacs = {}
	skipmacs = {}
	for loepnummer, mac in datasets.config.iterate(None, ('loepnummer', 'mac')):
		if loepnummer in unused_layers:
			skipmacs[loepnummer] = mac
		else:
			botmacs[loepnummer] = mac
	return botmacs, skipmacs
