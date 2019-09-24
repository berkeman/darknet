from extras import resolve_jobid_filename
import blob

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

#	jid = urd.build('complexity',           datasets=dict(source=jid_type))
#	jid = urd.build('reslayer_addmemsizes', datasets=dict(reslayers=jid + '/reslayers'))
#	jid = urd.build('reslayer_ccest',       datasets=dict(source=jid))
#	jid = urd.build('reslayer_printall',    datasets=dict(source=jid))


#	jid = urd.build('emlayer', jobids=dict(darknet=jid_darknet), options=dict(layers=64), datasets=dict(config=jid_type))

#	ev, reads, bdp = blob.load(jobid=jid)
#	print('sum error', sum(abs(x) for x in ev))
#	print('max error', max(abs(x) for x in ev))
#	print('num reads', reads)
#	print('bdp', bdp)




#	jid = urd.build('triplette', jobids=dict(darknet=jid_darknet), datasets=dict(config=jid_type))


	jid_bottlenecks = urd.build('findtriplettes', datasets=dict(config=jid_type))


	acost = []

	for c0size, c1size, c2size in (

		(112*1,100,30),
		(112*2,100,30),
		(112*3,100,30),
		(112*4,100,30),
		(112*5,100,30),

		(112*1,150,30),
		(112*2,150,30),
		(112*3,150,30),
		(112*4,150,30),
		(112*5,150,30),

		(112*1,125,30),
		(112*2,125,30),
		(112*3,125,30),
		(112*4,125,30),
		(112*5,125,30),

		(112*1,200,30),
		(112*2,200,30),
		(112*3,200,30),
		(112*4,200,30),
		(112*5,200,30),

		(112*1,400,30),
		(112*2,400,30),
		(112*3,400,30),
		(112*4,400,30),
		(112*5,400,30),
		(1000000,1000000,30),
	):

		jid = urd.build('triplette_new',
			jobids=dict(darknet=jid_darknet, bottlenecks=jid_bottlenecks),
			options=dict(cache0size=c0size, cache01size=c1size, cache12size=c2size)
		)
		res = blob.load(jobid=jid)

		xmemsize = 112*112*32

		print()
		print('=' * 80)
		print('cache sizes', c0size, c1size, c2size)
		print("                        ______   ____________________    ____________________    ____________________")
		print(" n  maxerr    SNR         X RD     0 RD   0HIT   0MIS     01 RD  01HIT  01MIS     12 RD  12HIT  12MIS")
		for t in sorted(res, key = lambda x: x['n']):
			t.cost = t.xrcnt*xmemsize + t.c0.reads*c0size + t.c1.reads*c1size + t.c2.reads*c2size
			print("%2d  %f  %5.2f    %6d   %6d %6d %6d    %6d %6d %6d    %6d %6d %6d  %9d %9d %9d %20s" % (
				t.n, t.maxerr, t.snr,
				t.xrcnt,
				t.c0.reads, t.c0.hits, t.c0.miss,
				t.c1.reads, t.c1.hits, t.c1.miss,
				t.c2.reads, t.c2.hits, t.c2.miss,
				t.cc0, t.cc1, t.cc2,
				"{:,}".format(t.cost),
			))
		print("\n")

		acost.append(res)


	for item in acost:
		print(item[0].c0size, item[0].c1size, item[0].c2size, item[0].xsize, sum(x.cost for x in item))



# @@@ varför funkar inte sista softmaxlagret?  Verkar inte som att all data sparas i convlayer*.c
# @@@ saknar 3x3 with stride

# @@@ kolla hur och varför det kraschar när cachen är "liten"

# @@@ räkna klockcykler tillsammans med block-aritmetiken på ngt vis, tänk på cache12 som kanske är nio disjunkta minnen!
