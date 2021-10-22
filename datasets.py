import datetime as dt
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle
from collections import defaultdict
import random
import glob

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm

import constants
import utils

dataset_objs     = {}
test_transforms  = {}
train_transforms = {}


#####################
# CIFAR 10 Dataset
#####################

class CIFAR10_base(datasets.CIFAR10):
	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
	This is a subclass of the `CIFAR10` Dataset.
	"""
	base_folder = 'cifar-10'
	url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	filename = "cifar-10-python.tar.gz"
	tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
	train_list = [
		['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
		['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
		['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
		['data_batch_4', '634d18415352ddfa80567beed471001a'],
		['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
	]

	test_list = [
		['test_batch', '40351d587109b95175f43aff81a1287e'],
	]
	meta = {
		'filename': 'batches.meta',
		'key': 'label_names',
		'md5': '5ff9c542aee3614f3951f8cda6e48888',
	}

class CIFAR10(Dataset):
	def __init__(self, root, train, transform, download=False):
		self.cifar10_base = CIFAR10_base(root=root,
										train=train,
										download=download,
										transform=transform)
		
	def __getitem__(self, index):
		data, target = self.cifar10_base[index]        
		return data, target, index

	def __len__(self):
		return len(self.cifar10_base)

dataset_objs['cifar10'] = CIFAR10

train_transforms['cifar10'] = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transforms['cifar10'] = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

##################
# CUB 200 Dataset
##################

class CUB2011(Dataset):
	base_folder = 'CUB_200_2011/images'
	url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
	filename = 'CUB_200_2011.tgz'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, train=True, transform=None, loader=default_loader):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.loader = default_loader
		self.train = train
		self._load_metadata()

	def _load_metadata(self):
		images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
							 names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
										 sep=' ', names=['img_id', 'target'])
		train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
									   sep=' ', names=['img_id', 'is_training_img'])

		data = images.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')
		
		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = sample.target - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)

		return img, target, idx

dataset_objs['cub200'] = CUB2011

train_transforms['cub200'] = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomResizedCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['cub200'] = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



####################
# NA Bird Dataset
####################

class NABird(Dataset):
	base_folder = 'images'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, train=True, transform=None, loader=default_loader):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.loader = default_loader
		self.train = train
		self._load_metadata()
		
	def _load_metadata(self):
		images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
							 names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(os.path.join(self.root, 'nabird_image_class.txt')),
										 sep=' ', names=['img_id', 'target'])
		train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
									   sep=' ', names=['img_id', 'is_training_img'])

		data = images.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')

		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = sample.target - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)

		return img, target, idx


dataset_objs['nabird'] = NABird

train_transforms['nabird'] = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomResizedCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['nabird'] = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


####################
# VOC2007 Dataset
####################

"""
adapted from Biagio Brattoli
"""

class VOC2007(Dataset):
	def __init__(self,data_path, train:bool, transform, loader=default_loader, random_crops=0):
		self.data_path = data_path
		self.transform = transform
		self.random_crops = random_crops
		self.trainval = 'train' if train else 'test'
		self.loader = default_loader
		self.__init_classes()
		self.names, self.labels = self.__dataset_info()
	
	def __getitem__(self, index):
		path = self.data_path + '/JPEGImages/'+self.names[index] + '.jpg'
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)

		y = self.labels[index]
		return img, y, index
	
	def __len__(self):
		return len(self.names)
	
	def __dataset_info(self):
		
		with open(self.data_path+'/ImageSets/Main/'+self.trainval+'.txt') as f:
			annotations = f.readlines()
		
		annotations = [n[:-1] for n in annotations]
		
		names  = []
		labels = []
		for af in annotations:
			if len(af)!=6:
				continue
			filename = os.path.join(self.data_path,'Annotations',af)
			tree = ET.parse(filename+'.xml')
			objs = tree.findall('object')
			num_objs = len(objs)
			
			boxes = np.zeros((num_objs, 4), dtype=np.uint16)
			boxes_cl = np.zeros((num_objs), dtype=np.int32)
			
			for ix, obj in enumerate(objs):               
				cls = self.class_to_ind[obj.find('name').text.lower().strip()]
				boxes_cl[ix] = cls
			
			lbl = np.zeros(self.num_classes)
			lbl[boxes_cl] = 1
			labels.append(lbl)
			names.append(af)
		
		return np.array(names), np.array(labels)#.astype(np.int_)
	
	def __init_classes(self):
		self.classes = ('__background__','aeroplane', 'bicycle', 'bird', 'boat',
						 'bottle', 'bus', 'car', 'cat', 'chair',
						 'cow', 'diningtable', 'dog', 'horse',
						 'motorbike', 'person', 'pottedplant',
						 'sheep', 'sofa', 'train', 'tvmonitor')
		self.num_classes  = len(self.classes)
		self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

dataset_objs['voc2007'] = VOC2007

train_transforms['voc2007'] = transforms.Compose([
	transforms.RandomResizedCrop(size=256),
	transforms.CenterCrop(size=224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['voc2007'] = transforms.Compose([
	transforms.Resize(size=256),
	transforms.CenterCrop(size=224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

########################
# Caltech 101 Dataset
########################

class Caltech101(Dataset):
	base_folder = '101_ObjectCategories'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, train=True, transform=None, loader=default_loader):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.loader = default_loader
		self.train = train
		self._load_metadata()

	def _load_metadata(self):
		images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
							 names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
										 sep=' ', names=['img_id', 'target'])
		train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
									   sep=' ', names=['img_id', 'is_training_img'])

		data = images.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')

		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = sample.target - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)

		return img, target, idx

dataset_objs['caltech101'] = Caltech101

train_transforms['caltech101'] = transforms.Compose([
	transforms.RandomResizedCrop(size=256),
	transforms.CenterCrop(size=224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['caltech101'] = transforms.Compose([
	transforms.Resize(size=256),
	transforms.CenterCrop(size=224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

##################
# Stanford Dogs
##################

class StanfordDogs(Dataset):
	"""`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
	Args:
		root (string): Root directory of dataset where directory
			``omniglot-py`` exists.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		download (bool, optional): If true, downloads the dataset tar files from the internet and
			puts it in root directory. If the tar files are already downloaded, they are not
			downloaded again.
	"""
	folder = 'StanfordDogs'
	download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

	def __init__(self,
				 root,
				 train=True,
				 transform=None,
				 download=False):

		self.root = os.path.join(os.path.expanduser(root), self.folder)
		self.train = train
		self.transform = transform
		if download:
			self.download()

		split = self.load_split()
		self.images_folder = os.path.join(self.root, 'Images')
		
		self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]
		self._flat_breed_images = self._breed_images

	def __len__(self):
		return len(self._flat_breed_images)

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target character class.
		"""
		image_name, target_class = self._flat_breed_images[index]
		image_path = os.path.join(self.images_folder, image_name)
		image = Image.open(image_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image, target_class, index

	def download(self):
		import tarfile

		if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
			if len(os.listdir(os.path.join(self.root, 'Images'))) == len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
				return

		for filename in ['images', 'annotation', 'lists']:
			tar_filename = filename + '.tar'
			url = self.download_url_prefix + '/' + tar_filename
			download_url(url, self.root, tar_filename, None)
			print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
			with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
				tar_file.extractall(self.root)
			os.remove(os.path.join(self.root, tar_filename))

	def load_split(self):
		if self.train:
			split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
			labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
		else:
			split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
			labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

		split = [item[0][0] for item in split]
		labels = [item[0]-1 for item in labels]
		return list(zip(split, labels))

dataset_objs['stanford_dogs'] = StanfordDogs

train_transforms['stanford_dogs'] = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomResizedCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['stanford_dogs'] = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


##################
# Oxford IIIT Dogs
##################

class OxfordPets(Dataset):
	"""`Oxford Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
	Args:
		root (string): Root directory of dataset where directory
			``omniglot-py`` exists.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		download (bool, optional): If true, downloads the dataset tar files from the internet and
			puts it in root directory. If the tar files are already downloaded, they are not
			downloaded again.
	"""
	folder = 'oxford_pets'

	def __init__(self,
				 root,
				 train=True,
				 transform=None,
				 loader=default_loader):

		self.root = os.path.join(os.path.expanduser(root), self.folder)
		self.train = train
		self.transform = transform
		self.loader = loader
		self._load_metadata()

	def __getitem__(self, idx):

		sample = self.data.iloc[idx]
		path = os.path.join(self.root, 'images', sample.img_id) + '.jpg'

		target = sample.class_id - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)

		return img, target, idx
	
	def _load_metadata(self):
		if self.train:
			train_file = os.path.join(self.root, 'annotations', 'trainval.txt')
			self.data = pd.read_csv(train_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])
		else:
			test_file = os.path.join(self.root, 'annotations', 'test.txt')
			self.data = pd.read_csv(test_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])

	def __len__(self):
		return len(self.data)

dataset_objs['oxford_pets'] = OxfordPets

train_transforms['oxford_pets'] = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomResizedCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['oxford_pets'] = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


####################
# Dataset Loader
####################



def construct_dataset(dataset:str, path:str, train:bool=False, **kwdargs) -> torch.utils.data.Dataset:
	# transform = (train_transforms[dataset] if train else test_transforms[dataset])
	transform = test_transforms[dataset] # Note: for training, use the above line. We're using the train set as the probe set, so use test transform
	return dataset_objs[dataset](path, train, transform=transform, **kwdargs)

def get_dataset_path(dataset:str) -> str:
	return f'./data/{dataset}/'


class ClassMapCache:
	""" Constructs and stores a cache of which instances map to which classes for each datset. """

	def __init__(self, dataset:str, train:bool):
		self.dataset = dataset
		self.train = train

		if not os.path.exists(self.cache_path):
			self.construct_cache()
		else:
			with open(self.cache_path, 'rb') as f:
				self.idx_to_class, self.class_to_idx = pickle.load(f)


	def construct_cache(self):
		print(f'Constructing class map for {self.dataset}...')
		dataset    = construct_dataset(self.dataset, get_dataset_path(self.dataset), self.train)
		dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

		self.idx_to_class = []
		self.class_to_idx = defaultdict(list)

		idx = 0

		for batch in tqdm(dataloader):
			y = batch[1]
			single_class = (y.ndim == 1)

			for _cls in y:
				if single_class:
					_cls = _cls.item()
				
				self.idx_to_class.append(_cls)
				
				if single_class:
					self.class_to_idx[_cls].append(idx)
				
				idx += 1
		
		self.class_to_idx = dict(self.class_to_idx)

		utils.make_dirs(self.cache_path)
		with open(self.cache_path, 'wb') as f:
			pickle.dump((self.idx_to_class, self.class_to_idx), f)



	@property
	def cache_path(self):
		return f'./cache/class_map/{self.dataset}_{"train" if self.train else "test"}.pkl'


class DatasetCache(torch.utils.data.Dataset):
	""" Constructs and stores a cache for the dataset post-transform. """

	def __init__(self, dataset:str, train:bool):
		self.dataset = dataset
		self.train = train

		self.cache_folder = os.path.split(self.cache_path(0))[0]
		
		if not os.path.exists(self.cache_path(0)):
			os.makedirs(self.cache_folder, exist_ok=True)
			self.construct_cache()
		
		self.length = len(glob.glob(self.glob_path()))
		self.class_map = ClassMapCache(dataset, train)
		
		super().__init__()

	def cache_path(self, idx:int) -> str:
		return f'./cache/datasets/{self.dataset}/{"train" if self.train else "test"}_{idx}.npy'
		
	def glob_path(self) -> str:
		return f'./cache/datasets/{self.dataset}/{"train" if self.train else "test"}_*'

	def construct_cache(self):
		print(f'Constructing dataset cache for {self.dataset}...')
		dataset    = construct_dataset(self.dataset, get_dataset_path(self.dataset), self.train)
		dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

		idx = 0

		for batch in tqdm(dataloader):
			x = batch[0]
			
			for i in range(x.shape[0]):
				np.save(self.cache_path(idx), x[i].numpy().astype(np.float16))
				idx += 1
	
	def __getitem__(self, idx:int) -> tuple:
		x = torch.from_numpy(np.load(self.cache_path(idx)).astype(np.float32))
		y = self.class_map.idx_to_class[idx]
		return x, y

	def __len__(self):
		return self.length


class BalancedClassSampler(torch.utils.data.DataLoader):
	""" Samples from a dataloader such that there's an equal number of instances per class. """

	def __init__(self, dataset:str, batch_size:int, instances_per_class:int, train:bool=True, **kwdargs):
		num_classes = constants.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train)
		map_cache = ClassMapCache(dataset, train)

		sampler_list = []

		for _, v in map_cache.class_to_idx.items():
			random.shuffle(v)
		
		for _ in range(instances_per_class):
			for i in range(num_classes):
				if i in map_cache.class_to_idx:
					idx_list = map_cache.class_to_idx[i]
					
					if len(idx_list) > 0:
						sampler_list.append(idx_list.pop())
		
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)


class FixedBudgetSampler(torch.utils.data.DataLoader):
	""" Samples from a dataloader such that there's a fixed number of samples. Classes are distributed evenly. """

	def __init__(self, dataset:str, batch_size:int, probe_size:int, train:bool=True, min_instances_per_class:int=2, **kwdargs):
		num_classes = constants.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train)
		map_cache = ClassMapCache(dataset, train)

		# VOC is multiclass so just sample a random subset
		if dataset == 'voc2007':
			samples = list(range(len(dataset_obj)))
			random.shuffle(samples)
			
			super().__init__(dataset_obj, batch_size, sampler=samples[:probe_size], **kwdargs)
			return

		sampler_list = []
		last_len = None

		for _, v in map_cache.class_to_idx.items():
			random.shuffle(v)
		
		class_indices = list(range(num_classes))
		class_indices = [i for i in class_indices if i in map_cache.class_to_idx] # Ensure that i exists

		# Whether or not to subsample the classes to meet the min_instances and probe_size quotas 
		if num_classes * min_instances_per_class > probe_size:
			# Randomly shuffle the classes so if we need to subsample the classes, it's random.
			random.shuffle(class_indices)
			# Select a subset of the classes to evaluate on.
			class_indices = class_indices[:probe_size // min_instances_per_class]
		
		# Updated the list of samples (sampler_list) each iteration with 1 image for each class
		# We stop when we're finished or there's a class we didn't add an image for (i.e., out of images).
		while last_len != len(sampler_list) and len(sampler_list) < probe_size:
			# This is to ensure we don't infinitely loop if we run out of images
			last_len = len(sampler_list)

			for i in class_indices:
				idx_list = map_cache.class_to_idx[i]
				
				# If we still have images left of this class
				if len(idx_list) > 0:
					# Add it to the list of samples
					sampler_list.append(idx_list.pop())
				
				if len(sampler_list) >= probe_size:
					break
		
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)
		

if __name__ == '__main__':
	FixedBudgetSampler('voc2007', 128, 500, train=True)
