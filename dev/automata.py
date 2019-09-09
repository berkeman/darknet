
def main(urd):

#./darknet classifier predict cfg/imagenet1k.data mobilenet/test.cfg  mobilenet/test.weights mobilenet/cat.jpg | head -54 > kalle

	jid = urd.build('csvimport',
		options=dict(
			filename='nisse',
			separator=',',
			labelsonfirstline=True,
			allow_bad=False,
#			labels=(
#				'type',
#				'n', 'sx', 'sy', 'stride',
#				'wi', 'hi', 'ci',
#				'wo', 'ho', 'co',
#				'macs',
#			),
		)
	)
	jid_type = urd.build('dataset_type',
		datasets=dict(source=jid),
		options=dict(
			column2type=dict(
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
