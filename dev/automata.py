from collections import Counter

from accelerator.extras import resolve_jobid_filename
from accelerator import blob

bitsperword = 8
WL = 32
if WL>=24:
	m0size = 224*224
else:
	print("fix m0size, min requirement is 224*224*24 words")
	exit()

system_freq = 1e9

def main(urd):

	jid_darknet = urd.build('darknet')

	jid = urd.build('csvimport',
		options=dict(
			filename=resolve_jobid_filename(jid_darknet, 'configuration.txt'),
			separator=',',
			labelsonfirstline=True,
			allow_bad=False,
		)
	)
	jid_type = urd.build('dataset_type',
		datasets=dict(source=jid),
		options=dict(
			column2type=dict(
				loepnummer='number',
				layer='unicode:UTF-8',
				n='number',
				sx='number',
				sy='number',
				groups='number',
				stride='number',
				wi='number',
				hi='number',
				ci='number',
				wo='number',
				ho='number',
				co='number',
				mac='number',
			),
		)
	)

	jid_bottlenecks = urd.build('findtriplettes', datasets=dict(config=jid_type))

#	jid = urd.build('teststride', jobids=dict(darknet=jid_darknet))

	urd.build('addmemsizes', datasets=dict(config=jid_type))

	jid_macs = urd.build('countmacs',
		jobids=dict(bottlenecks=jid_bottlenecks),
		datasets=dict(config=jid_type),
	)
	x = blob.load(jobid=jid_macs)
	botmacs = sum(x[0].values())
	skipmacs = sum(x[1].values())
	print('MACs in bottleneck layers', botmacs)
	print('MACs in skipped layers   ', skipmacs)
	print('skipped ratio            ', skipmacs / (skipmacs + botmacs))



	costs = []

	for c0size, c1size, c2size in (

		(112*1,10000,30),
		(112*2,10000,30),
		(112*3,10000,30),
		(112*4,10000,30),

		(112*4,1000,30),
		(112*4,500,30),
		(112*4,400,30),
		(112*4,300,30),
		(112*4,200,30),

		(112*1,400,30),

		(m0size, m0size, m0size),
		(1e6, 1e6, 1e6),

#		(1e6, m0size, m0size),
#		(m0size, 1e6, m0size),
#		(m0size, m0size, 1e6),
	):

		def cacti(size):
			jid = urd.build('cacti', options=dict(args=dict(C=size, OUTPUT_WIDTH=32*8, B=32)))
			return blob.load(jobid=jid)['Dynamic read energy (nJ)']

#		costm = cacti(m0size)
#		cost0 = cacti(c0size)
#		cost1 = cacti(c1size)
#		cost2 = cacti(c2size)

		costm = m0size
		cost0 = c0size
		cost1 = c1size
		cost2 = c2size

		MUL1 = 1
		MUL3 = 1


		def colmag(f):
			if f >= 3.0:
				return '[31m%6.2f[0m' % (f,)
			if f >= 2.0:
				return '[35m%6.2f[0m' % (f,)
			if f > 1.0:
				return '[36m%6.2f[0m' % (f,)
			else:
				return '[37m%6.2f[0m' % (f,)

		jid = urd.build('bottleneck',
			jobids=dict(darknet=jid_darknet, bottlenecks=jid_bottlenecks),
			options=dict(xmemsize=m0size, cache0size=c0size, cache01size=c1size, cache12size=c2size, WL=WL, runonly=None)
		)
		res = blob.load(jobid=jid)
		tot = Counter()
		print()
		print('=' * 80)
		print('cache sizes', c0size, c1size, c2size)
		print('#MULS = %4d  %4d  %4d' % (WL*MUL1, WL*MUL3, WL*MUL1))
		print("                        ______   ____________________    ____________________    ____________________    _______________________       ________________     ________________")
		print(" n  maxerr    SNR         X RD     0 RD   0HIT   0MIS     01 RD  01HIT  01MIS     12 RD  12HIT  12MIS     cc 1x1  cc 3x3  cc 1x1             tot energy     tot clock cycles")

		for t in sorted(res, key = lambda x: x['n']):
			t.energy = t.xrcnt*costm + t.c0.reads*cost0 + t.c1.reads*cost1 + t.c2.reads*cost2
			t.l0stat['cc'] /= MUL1
			t.l1stat['cc'] /= MUL3
			t.l2stat['cc'] /= MUL1
			t.cc = t.l0stat['cc'] + t.l1stat['cc'] + t.l2stat['cc']
			print("%2d  %f  %5.2f    %6d   %6d %6d %6d    %6d %6d %6d    %6d %6d %6d    %7d %7d %7d   %20s %20s   %s" % (
				t.n, t.maxerr, t.snr,
				t.xrcnt,
				t.c0.reads, t.c0.hits, t.c0.miss,
				t.c1.reads, t.c1.hits, t.c1.miss,
				t.c2.reads, t.c2.hits, t.c2.miss,
				t.l0stat['cc'], t.l1stat['cc'], t.l2stat['cc'],
				"{:,}".format(t.energy),
				"{:,}".format(t.cc),
				"%dx%d %d->%d->%d %dx%d" % (t.l0stat['wi'], t.l0stat['hi'], t.l0stat['ci'], t.l2stat['ci'], t.l2stat['co'], t.l2stat['wo'], t.l2stat['ho'],),
			))
			tot['cc'] += t.cc
			tot['e']  += t.energy
		print("                                                                                                                                   %20s %20s" % (
			"{:,}".format(tot['e']),
			"{:,}".format(tot['cc']),
		))
		print()
		print("                                                                                                                                                            layer      featin   featout")
		for t in sorted(res, key = lambda x: x['n']):
			t.energy = t.xrcnt*costm + t.c0.reads*cost0 + t.c1.reads*cost1 + t.c2.reads*cost2
			t.l0stat['cc'] /= MUL1
			t.l1stat['cc'] /= MUL3
			t.l2stat['cc'] /= MUL1
			t.cc = t.l0stat['cc'] + t.l1stat['cc'] + t.l2stat['cc']
			print("                        %6s                 %6s                  %6s                  %6s     %6s  %6s  %6s   %30s   %9d %9d" % (
				colmag(t.xrcnt   / (t.l0stat['wi']*t.l0stat['hi']*t.l0stat['ci']/t.l0stat['WL'])),
				colmag(t.c0.miss / (t.l0stat['wi']*t.l0stat['hi'])),
				colmag(t.c1.miss / (t.l1stat['wi']*t.l1stat['hi'])),
				colmag(t.c2.miss / (t.l2stat['wi']*t.l2stat['hi'])),
				colmag(MUL1 * WL * t.l0stat['cc'] / (t.l0stat['wi']*t.l0stat['hi']*t.l0stat['ci']*t.l0stat['co'])),
				colmag(MUL3 * WL * t.l1stat['cc'] / (t.l1stat['wo']*t.l1stat['ho']*t.l1stat['co']* 3 * 3)),
				colmag(MUL1 * WL * t.l2stat['cc'] / (t.l2stat['wi']*t.l2stat['hi']*t.l2stat['ci']*t.l2stat['co'])),
				"%dx%d %d->%d->%d %dx%d" % (t.l0stat['wi'], t.l0stat['hi'], t.l0stat['ci'], t.l2stat['ci'], t.l2stat['co'], t.l2stat['wo'], t.l2stat['ho'],),
				t.l0stat['wi']*t.l0stat['hi']*t.l0stat['ci'],
				t.l1stat['wi']*t.l1stat['hi']*t.l1stat['ci'],
			))
		print("\n")


		costs.append(res)

	print("     xmem        c0         c1         c2           energy           cc      rate")
	for item in costs:
		print("%9d %10d %10d %10d    %12d %12d  %5.2ffps" % (
			item[0].xsize*WL*bitsperword,
			item[0].c0size*WL*bitsperword,
			item[0].c1size*WL*bitsperword,
			item[0].c2size*WL*bitsperword,
			sum(x.energy for x in item),
			sum(x.cc for x in item),
			system_freq/sum(x.cc for x in item)
		))
	print("All memory sizes in *bits*")

	from accelerator.automata_common import profile_jobs
	print('\nExec time', profile_jobs(urd.joblist))
