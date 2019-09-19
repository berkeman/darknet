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

	jid = urd.build('complexity',           datasets=dict(source=jid_type))
	jid = urd.build('reslayer_addmemsizes', datasets=dict(reslayers=jid + '/reslayers'))
	jid = urd.build('reslayer_ccest',       datasets=dict(source=jid))
	jid = urd.build('reslayer_printall',    datasets=dict(source=jid))


	jid = urd.build('emlayer', jobids=dict(darknet=jid_darknet), options=dict(layers=64), datasets=dict(config=jid_type))

	ev, reads, bdp = blob.load(jobid=jid)
	print('sum error', sum(abs(x) for x in ev))
	print('max error', max(abs(x) for x in ev))
	print('num reads', reads)
	print('bdp', bdp)




	jid = urd.build('triplette', jobids=dict(darknet=jid_darknet), datasets=dict(config=jid_type))

	jid = urd.build('triplette_new', jobids=dict(darknet=jid_darknet), datasets=dict(config=jid_type))
	res = blob.load(jobid=jid)

	print()
	print("                        ------   --------------------    --------------------    --------------------")
	print(" n  maxerr    SNR         X RD     0 RD   0HIT   0MIS     01 RD  01HIT  01MIS     12 RD  12HIT  12MIS")
	for item in res:
		print("%2d  %f  %5.2f    %6d   %6d %6d %6d    %6d %6d %6d    %6d %6d %6d" % (item[0], item[1], item[2], item[6], item[3][0], item[3][1], item[3][2], item[4][0], item[4][1], item[4][2], item[5][0], item[5][1], item[5][2], ))
	print("\n")

# @@@ varför funkar inte sista softmaxlagret?  Verkar inte som att all data sparas i convlayer*.c
# @@@ saknar 3x3 with stride

# @@@ kolla hur och varför det kraschar när cachen är "liten"

# @@@ räkna klockcykler tillsammans med block-aritmetiken på ngt vis, tänk på cache12 som kanske är nio disjunkta minnen!
