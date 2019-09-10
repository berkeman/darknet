
def main(urd):

#./darknet classifier predict cfg/imagenet1k.data mobilenet/test.cfg  mobilenet/test.weights mobilenet/cat.jpg | head -54 > kalle

	jid = urd.build('csvimport',
		options=dict(
			filename='mobilenetv2_formatted_conf.txt',
			separator=',',
			labelsonfirstline=True,
			allow_bad=False,
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
	jid = urd.build('reslayer_printall',    datasets=dict(source=jid))
