from collections import Counter

from extras import resolve_jobid_filename
import blob

WL = 32
m0size = 112*112*3


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

	acost = []

	for c0size, c1size, c2size in (

		(112*4,400,30),# (112*3,400,30), (112*2.5,400,30), (112*2,400,30), (112*1,400,30),
		(112*4,300,30),# (112*3,300,30), (112*2.5,300,30), (112*2,300,30), (112*1,300,30),
#		(112*4,200,30), (112*3,200,30), (112*2.5,200,30), (112*2,200,30), (112*1,200,30),

		(m0size, m0size, m0size),
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


		jid_macs = urd.build('countmacs', jobids=dict(darknet=jid_darknet, bottlenecks=jid_bottlenecks))


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
		print("                        ______   ____________________    ____________________    ____________________    _______________________       ________________     ________________")
		print(" n  maxerr    SNR         X RD     0 RD   0HIT   0MIS     01 RD  01HIT  01MIS     12 RD  12HIT  12MIS     cc 1x1  cc 3x3  cc 1x1             tot energy     tot clock cycles")
		for t in sorted(res, key = lambda x: x['n']):
			t.energy = t.xrcnt*costm + t.c0.reads*cost0 + t.c1.reads*cost1 + t.c2.reads*cost2
			t.cc = t.l0stat['cc'] + t.l1stat['cc'] + t.l2stat['cc']
			print("%2d  %f  %5.2f    %6d   %6d %6d %6d    %6d %6d %6d    %6d %6d %6d    %7d %7d %7d   %20s %20s" % (
				t.n, t.maxerr, t.snr,
				t.xrcnt,
				t.c0.reads, t.c0.hits, t.c0.miss,
				t.c1.reads, t.c1.hits, t.c1.miss,
				t.c2.reads, t.c2.hits, t.c2.miss,
				t.l0stat['cc'], t.l1stat['cc'], t.l2stat['cc'],
				"{:,}".format(t.energy),
				"{:,}".format(t.cc),
			))

			print("                        %6s                 %6s                  %6s                  %6s     %6s  %6s  %6s" % (
				colmag(t.xrcnt / (t.l0stat['wi']*t.l0stat['hi']*t.l0stat['ci']/t.l0stat['WL'])),
				colmag(t.c0.miss / (t.l0stat['wi']*t.l0stat['hi'])),
				colmag(t.c1.miss / (t.l1stat['wi']*t.l1stat['hi'])),
				colmag(t.c2.miss / (t.l2stat['wi']*t.l2stat['hi'])),
				colmag(WL * t.l0stat['cc'] / (t.l0stat['wi']*t.l0stat['hi']*t.l0stat['ci']*t.l0stat['co'])),
				colmag(3*3*WL * t.l1stat['cc'] / (t.l1stat['wo']*t.l1stat['ho']*t.l1stat['co']* 3 * 3)),
				colmag(WL * t.l2stat['cc'] / (t.l2stat['wi']*t.l2stat['hi']*t.l2stat['ci']*t.l2stat['co'])),
			))

			tot['cc'] += t.cc
			tot['e']  += t.energy
		print("                                                                                                                                   %20s %20s" % (
			"{:,}".format(tot['e']),
			"{:,}".format(tot['cc']),
		))
		print("\n")
		acost.append(res)




	for item in acost:
		print("%9d %9d %9d %9d    %12d %12d" % (item[0].xsize, item[0].c0size, item[0].c1size, item[0].c2size, sum(x.energy for x in item), sum(x.cc for x in item)))

	from automata_common import profile_jobs
	print('Exec time', profile_jobs(urd.joblist))



# @@@ varför funkar inte sista softmaxlagret?  Verkar inte som att all data sparas i convlayer*.c
# @@@ saknar 3x3 with stride

# @@@ kolla hur och varför det kraschar när cachen är "liten"

# @@@ räkna klockcykler tillsammans med block-aritmetiken på ngt vis, tänk på cache12 som kanske är nio disjunkta minnen!
