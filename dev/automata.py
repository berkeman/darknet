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




# @@@ kör darknet via metod
# @@@ se till att resten läser all data från denna.
