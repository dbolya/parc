import torchvision
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import pickle
import itertools
import time
import gc
import csv
from collections import defaultdict
import argparse
from typing import Dict

from constants import variables, num_classes, external
from datasets import BalancedClassSampler, FixedBudgetSampler
import utils
from methods import TransferabilityMethod


class ClassBalancedExperimentParams:
	""" Using a fixed number of instances per class. """

	def __init__(self, instances_per_class:int):
		self.instances_per_class = instances_per_class
		self.experiment_name = f'class_balanced_{self.instances_per_class}'

	def create_dataloader(self, dataset:str, batch_size:int, **kwdargs):
		return BalancedClassSampler(dataset, batch_size=128, instances_per_class=self.instances_per_class, **kwdargs)
		

class FixedBudgetExperimentParams:
	""" Using a fixed budget probe size with classes distributed as evenly as possible. """

	def __init__(self, budget:int):
		self.budget = budget
		self.experiment_name = f'fixed_budget_{self.budget}'

	def create_dataloader(self, dataset:str, batch_size:int, **kwdargs):
		return FixedBudgetSampler(dataset, batch_size=128, probe_size=self.budget, **kwdargs)


class Experiment:

	"""
	Runs the given methosd on each probe set and outputs score and timing information into ./results/.
	To evaluate the results, see metrics.py.

	Params:
	 - methods: A dictionary of methods to use.
	 - budget: The number of images in each probe set. Leave as default unless you want to extract your own probe sets.
	 - runs: The number of different probes sampled per transfer. Leave as default unless you want to extract your own probe sets.
	 - probe_only: If True, skips doing method computation and instead only extracts the probe sets.
	 - model_bank: Which model bank to use. Options are: "controlled" (default) and "all" (includes crowd-sourced).
	 - append: If false (default), the output file will be overwritten. Otherwise, it will resume from where it left off. When resuming, timing information will be lost.
	 - name: The name of the experiment. Defaults to the name of the probe set.
	"""

	def __init__(self, methods:Dict[str, TransferabilityMethod], budget:int=500, runs:int=5, probe_only:bool=False, model_bank:str='controlled', append:bool=False, name:str=None):
		self.params = FixedBudgetExperimentParams(budget)
		self.runs = runs
		self.probe_only = probe_only
		self.model_bank = model_bank
		self.name = name if name is not None else self.params.experiment_name
		self.methods = methods

		self.dataloaders = {}

		key = ['Run', 'Architecture', 'Source Dataset', 'Target Dataset']
		headers = key + list(self.methods.keys())

		self.out_cache = utils.CSVCache(self.out_file, headers, key=key, append=append)

		self.times = defaultdict(list)
		
	def cache_path(self, architecture:str, source_dataset:str, target_dataset:str, run:int):
		return f'./cache/probes/{self.params.experiment_name}/{architecture}_{source_dataset}_{target_dataset}_{run}.pkl'

	@property
	def cur_cache_path(self):
		return self.cache_path(self.architecture, self.source_dataset, self.target_dataset, self.run)

	@property
	def out_file(self):
		return f'./results/{self.name}.csv'

	@property
	def timing_file(self):
		return f'./results/{self.name}_timing.pkl'

	def prep_model(self):
		model = utils.load_source_model(self.architecture, self.source_dataset)
		model.cuda()
		model.eval()

		def extract_feats(self, args):
			x = args[0]
			model._extracted_feats[x.get_device()] = x

		for name, module in model.named_modules():
			if isinstance(module, nn.Linear):
				module.register_forward_pre_hook(extract_feats)
		
		return model

	def probe(self):
		""" Returns (and creates if necessary) probe data for the current run. """
		cache_path = self.cur_cache_path
		
		if os.path.exists(cache_path):
			with open(cache_path, 'rb') as f:
				return pickle.load(f)
		
		if self.model == None:
			self.model = self.prep_model()

		dataloader_key = (self.target_dataset, self.run)

		if dataloader_key not in self.dataloaders:
			utils.seed_all(2020 + self.run * 3037)
			dataloader = self.params.create_dataloader(self.target_dataset, batch_size=128, train=True, pin_memory=True)
			self.dataloaders[dataloader_key] = dataloader
		dataloader = self.dataloaders[dataloader_key]

		with torch.no_grad():
			all_y     = []
			all_feats = []
			all_probs = []

			for x, y in dataloader:
				# Support for using multiple GPUs
				self.model._extracted_feats = [None] * torch.cuda.device_count()

				x = x.cuda()
				preds = self.model(x)

				all_y.append(y.cpu())
				all_probs.append(torch.nn.functional.softmax(preds, dim=-1).cpu())
				all_feats.append(torch.cat([x.cpu() for x in self.model._extracted_feats], dim=0))

			all_y     = torch.cat(all_y    , dim=0).numpy()
			all_feats = torch.cat(all_feats, dim=0).numpy()
			all_probs = torch.cat(all_probs, dim=0).numpy()

			params = {
				'features': all_feats,
				'probs': all_probs,
				'y': all_y,
				'source_dataset': self.source_dataset,
				'target_dataset': self.target_dataset,
				'architecture': self.architecture
			}
		
		utils.make_dirs(cache_path)
		with open(cache_path, 'wb') as f:
			pickle.dump(params, f)
		
		return params

	def evaluate(self):
		params = self.probe()

		if self.probe_only:
			return

		if self.source_dataset == self.target_dataset:
			return

		params['cache_path_fn'] = lambda architecture, source, target: self.cache_path(architecture, source, target, self.run)
		
		scores = [self.run, self.architecture, self.source_dataset, self.target_dataset]

		for idx, (name, method) in enumerate(self.methods.items()):
			utils.seed_all(1010 + self.run * 2131)
			last_time = time.time()
			scores.append(method(**params))
			self.times[name].append(time.time() - last_time)

		self.out_cache.write_row(scores)

		
			

	def run(self):
		""" Run the methods on the data and then saves it to out_path. """
		last_model = None

		factors = [variables['Architecture'], variables['Source Dataset'], variables['Target Dataset'], list(range(self.runs))]

		iter_obj = []		

		if self.model_bank == 'all':
			for arch, source in external:
				for target in variables['Target Dataset']:
					for run in range(self.runs):
						iter_obj.append((arch, source, target, run))

		iter_obj += list(itertools.product(*factors))

		for arch, source, target, run in tqdm(iter_obj):
			# RSA requires source-source extraction, so keep this out
			# if source == target:
			# 	continue

			if self.out_cache.exists(run, arch, source, target):
				continue

			cur_model = (arch, source)
			if cur_model != last_model:
				self.model = None

			self.architecture = arch
			self.source_dataset = source
			self.target_dataset = target
			self.run = run

			self.evaluate()

			gc.collect()
		
		for name, times in self.times.items():
			print(f'{name:20s}: {sum(times) / len(times): .3f}s average')
		
		with open(self.timing_file, 'wb') as f:
			pickle.dump(dict(self.times), f)
		

		

		


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--budget'    , help='Number of image in the probe set. Default is 500.', default=500, type=int)
	parser.add_argument('--runs'      , help='Number of probe sets sampled per transfer. Default is 5.', default=5, type=int)
	parser.add_argument('--probe_only', help='Set this flag if you only want to generate probe sets.', action='store_true')
	parser.add_argument('--model_bank', help='Which model bank to use. Options are "controlled" and "all". Default is "controlled".', default='controlled', type=str)
	args = parser.parse_args()

	Experiment(args.budget, runs=args.runs, probe_only=args.probe_only, model_bank=args.model_bank).run()


