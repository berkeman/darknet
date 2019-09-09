import blob

jobids=('source',)

def synthesis():

	reslayers, remlayers = blob.load(jobid=jobids.source)

	for item in reslayers:
		print()
		print("%dx%d-%d -> %dx%d-%d -> %dx%d-%d" % (
			item.wi, item.hi, item.ci,
			item.wi, item.hi, item.cm,
			item.wo, item.ho, item.co,
		))
		inputfeature  = item.wi * item.hi * item.ci
		twolinechache = item.wi * 2 * item.ci
		dwcache9      = 9* item.cm
		print(inputfeature, twolinechache, dwcache9)
