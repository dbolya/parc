variables = {
	'Source Dataset' : [
		'nabird',
		'oxford_pets',
		'cub200',
		'caltech101',
		'stanford_dogs',
		'voc2007',
		'cifar10',
		'imagenet'
	],

	'Target Dataset' : [
		'cifar10',
		'oxford_pets',
		'cub200',
		'caltech101',
		'stanford_dogs',
		'nabird',
# 		'voc2007'  # Only some methods support multi-label
	],

	'Architecture': [
		'resnet50',
		'resnet18',
		'googlenet',
		'alexnet',
	]
}

num_classes = {
	'nabird': 555,
	'oxford_pets': 37,
	'cub200': 200,
	'caltech101': 101,
	'stanford_dogs': 120,
	'voc2007': 21,
	'cifar10': 10,
	'imagenet': 1000
}

dataset_images = {
    'nabird': 24633,
    'oxford_pets': 3669,
    'cub200': 5794,
    'caltech101': 2630,
    'stanford_dogs': 8580,
    'voc2007': 2501,
    'cifar10': 50000,
    'imagenet': 150000
}

model_layers = {
  'resnet50': 50,
  'resnet18': 18,
  'googlenet': 22,
  'alexnet': 8,
  'resnet101': 101
}

model_banks = [
	'controlled',
	'all'
]

external = {
	('resnet101', 'faster-rcnn_c4_138204752'),
	('resnet101', 'faster-rcnn_fpn_137851257'),
	('resnet101', 'keypoint-rcnn_fpn_138363331'),
	('resnet101', 'mask-rcnn_c4_138363239'),
	('resnet101', 'mask-rcnn_fpn_138205316'),
	('resnet101', 'mask-rcnn_fpn_lvis_144219035'),
	('resnet101', 'panoptic-rcnn_fpn_139514519'),
	('resnet101', 'panoptic-rcnn_fpn_139797668'),
	('resnet101', 'retinanet_190397697'),
	('resnet101', 'simclr_imagenet'),
	('resnet50', 'clusterfit_imagenet'),
	('resnet50', 'deepclusterv2_imagenet'),
	('resnet50', 'faster-rcnn_c4_137257644'),
	('resnet50', 'faster-rcnn_c4_137849393'),
	('resnet50', 'faster-rcnn_c4_voc_142202221'),
	('resnet50', 'faster-rcnn_fpn_137257794'),
	('resnet50', 'faster-rcnn_fpn_137849458'),
	('resnet50', 'jigsaw_imagenet22k'),
	('resnet50', 'keypoint-rcnn_fpn_137261548'),
	('resnet50', 'keypoint-rcnn_fpn_137849621'),
	('resnet50', 'mask-rcnn_c4_137259246'),
	('resnet50', 'mask-rcnn_c4_137849525'),
	('resnet50', 'mask-rcnn_fpn_137260431'),
	('resnet50', 'mask-rcnn_fpn_137849600'),
	('resnet50', 'mask-rcnn_fpn_cityscapes_142423278'),
	('resnet50', 'mask-rcnn_fpn_lvis_144219072'),
	('resnet50', 'moco_imagenet'),
	('resnet50', 'npid_imagenet'),
	('resnet50', 'panoptic-rcnn_fpn_139514544'),
	('resnet50', 'panoptic-rcnn_fpn_139514569'),
	('resnet50', 'pirl_imagenet'),
	('resnet50', 'retinanet_190397773'),
	('resnet50', 'retinanet_190397829'),
	('resnet50', 'rotnet_imagenet22k'),
	('resnet50', 'semisup_instagram'),
	('resnet50', 'semisup_yfcc100m'),
	('resnet50', 'simclr_imagenet'),
	('resnet50', 'supervised_places205'),
	('resnet50', 'swav_imagenet'),
}
