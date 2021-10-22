import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import pickle
from collections import defaultdict

from methods import LearnedHeuristic
import constants

def add_oracle(results:pd.DataFrame, oracle_path:str):
	oracle_scores = []
	oracle_csv = pd.read_csv(oracle_path)

	for idx, row in results.iterrows():
		x = oracle_csv
		x = x[x['Architecture'] == row['Architecture']]
		x = x[x['Source Dataset'] == row['Source Dataset']]
		x = x[x['Target Dataset'] == row['Target Dataset']]
		x = x['Oracle']

		if row['Source Dataset'] in ('swav_imagenet', 'deepclusterv2_imagenet', 'npid_imagenet', 'moco_imagenet', 'simclr_imagenet'):
			x = -1
		oracle_scores.append(float(x))

	results.insert(len(results.columns), 'Oracle', oracle_scores)
	return results

def add_heuristic(results:pd.DataFrame):	
	intuition_scores = []
	lh_scores = []

	lh = LearnedHeuristic()
	lh.load()

	for idx, row in results.iterrows():
		model = row['Architecture']
		dataset = row['Source Dataset']
		target = row['Target Dataset']

		intuition_score = constants.model_layers[model] + np.log(constants.dataset_images[dataset]) - np.log(constants.dataset_images[target])
		intuition_scores.append(intuition_score)

		lh_score = lh.predict(lh.make_feature(model, dataset, target))
		lh_scores.append(lh_score)

	results.insert(len(results.columns), 'Heuristic', intuition_scores)
	results.insert(len(results.columns), 'Learned Heuristic', lh_scores)
	return results

def add_plasticity(results:pd.DataFrame, methods:list, weight:float=1) -> pd.DataFrame:
	"""
		Accounts for model plasticity by ensembling each method with
		a heuristic based on the # of layers in the source model.

		Scores for each method are normalized first so that the weight
		on the heuristic is consistent.
	"""
	stats = {}

	for method in methods:
		col = results[method]
		stats[method] = (col.mean(), col.min(), col.max(), col.std())
	
	for idx, row in results.iterrows():
		layers = constants.model_layers[row['Architecture']]

		for method in methods:
			val = results.at[idx, method]
			mean, cmin, cmax, std = stats[method]

			val = (val - mean) / std
			# val = (val - cmin) / (cmax - cmin)

			results.at[idx, method] = val + weight * (layers / 50)
	
	return results


def pearson(method:np.ndarray, gt:np.ndarray):
	idx = (gt > 0)
	method = method[idx]
	gt = gt[idx]

	# import matplotlib.pyplot as plt
	# plt.figure()
	# plt.plot(method, gt, 'o')
	# plt.show()

	return stats.pearsonr(method, gt)[0] * 100



class MetricEval:
	"""
	Evaluates the csv created by evaluate.py against the oracle transfer performance.

	Params:
		- path: The path to the csv you want to evaluate.
		- oracle_path: The path to the oracle file. This depends on which benchmark you are using.
	"""

	def __init__(self, path:str, oracle_path:str='./oracles/controlled.csv'):
		self.results = pd.read_csv(path)
		self.methods = list(self.results.columns)[4:]

		# self.results = add_heuristic(self.results)
		self.results = add_oracle(self.results, oracle_path)

		self.results = self.results.fillna(0)
	
	def all_methods(self, ignore_methods:set=None):
		methods = self.methods # + ['Heuristic', 'Learned Heuristic']
		if ignore_methods is not None:
			for method in ignore_methods:
				methods.remove(methods.index(method))
		return methods

	def add_plasticity(self, methods:list=None, weight:int=1):
		if methods is None:
			methods = self.all_methods()

		add_plasticity(self.results, methods, weight)

	def aggregate(self, constants:list=['Target Dataset'], methods:list=None, variance_over:str='Run', metric=pearson, aggregate:bool=True):
		"""
		Aggregates the results for the given methods (default all) and parameters.

		Params:
			- constants: A list of variables names to keep constant when evaluating. Constants will be averaged over, while correlation is computed over everything not constant.
			- methods:   The subset of methods to evaluate. This defaults to all methods.
			- variance_over: The csv column to compute variance over. Probably don't want to change this.
			- metric: The metric function to use. See the definition of pearson for more details.
			- aggregate: Whether or not to average over all costants. If False, this will return a separate result for every constant.

		Returns mean, variance, all_runs.
		"""
		all_runs = []

		constant_values = {
			k: list(self.results[k].unique())
			for k in constants
		}

		if methods is None:
			methods = self.all_methods()

		for run in self.results[variance_over].unique():
			# Select the current run
			run_results = self.results[self.results[variance_over] == run]
			run_metrics = {}

			# Set up the iterables for filtering
			if len(constants) == 0:
				iter_obj = [None]
			else:
				iter_obj = itertools.product(*[constant_values[k] for k in constants])

			# Now iterate through all the possible combination of constant values
			for cur_values in iter_obj:
				cur_results = run_results
				cur_metrics = {}

				# Do the filtering
				if cur_values != None:
					for constant, value in zip(constants, cur_values):
						cur_results = cur_results[cur_results[constant] == value]
					
					cur_values = tuple(cur_values)
				
				if len(cur_results) < 2:
					continue

				# Do evaluation
				gt_results = cur_results['Oracle'].to_numpy()

				# if cur_values[0] != 'voc2007':
				# 	continue
				# print(cur_results[cur_results['Oracle'] > 96])

				for method in methods:
					# print(cur_values, method)
					method_results = cur_results[method].to_numpy()
					method_score   = metric(method_results, gt_results)
					cur_metrics[method] = method_score
				
				run_metrics[cur_values] = cur_metrics
			

			# Aggregate the results
			if aggregate:
				aggregated_run_metrics = {k: 0 for k in methods}
				total = 0

				# Sum all of the metric results for this run over all possibilities for constant values
				for _, cur_metrics in run_metrics.items():
					total += 1
					for k, v in cur_metrics.items():
						aggregated_run_metrics[k] += v
				
				# Turn the sums into means
				for method in methods:
					aggregated_run_metrics[method] /= total

				all_runs.append(aggregated_run_metrics)
			else:
				all_runs.append(run_metrics)

		

		def compute_variance(run_data:list, mean:float):
			return sum([(x - mean) ** 2 for x in run_data]) / len(run_data)


		# Compute variance
		if aggregate:
			run_data = {method: [run[method] for run in all_runs] for method in methods}

			mean     = {method: sum(run_data[method]) / len(all_runs) for method in methods}
			variance = {method: compute_variance(run_data[method], mean[method]) for method in methods}

			return mean, variance, all_runs
		else:
			run_data = {
				possibility: {
					method: [run[possibility][method] for run in all_runs]
					for method in methods
				} for possibility in all_runs[0]
			}

			mean = {
				possibility: {
					method: sum(run_data[possibility][method]) / len(all_runs)
					for method in methods
				} for possibility in all_runs[0]
			}

			variance = {
				possibility: {
					method: compute_variance(run_data[possibility][method], mean[possibility][method])
					for method in methods
				} for possibility in all_runs[0]
			}

			return mean, variance, all_runs




if __name__ == '__main__':
	# path = './results/data_sweep/fixed_budget_500.csv'
	path = './results/fixed_budget_500.csv'
	eval_obj = MetricEval(path)
	eval_obj.add_plasticity()
	constants = ['Target Dataset'] # ['Source Dataset', 'Architecture'] # 

	with open(path.replace('.csv', '_timing.pkl'), 'rb') as f:
		timing = pickle.load(f)

	# mean, var, _ = eval_obj.aggregate(constants=constants, aggregate=False)
	# for possibility in mean:
	# 	print(f' --- {possibility} --- ')
	# 	for method in mean[possibility]:
	# 		print(f'{method:20s}: {mean[possibility][method]:6.2f}% +/- {np.sqrt(var[possibility][method]):4.2f}')
	# 	print()
	
	mean, var, _ = eval_obj.aggregate(constants=constants, aggregate=True)
	print(' --- TOTAL --- ')
	for method in mean:
		avg_time = sum(timing[method]) / len(timing[method]) if method in timing else 0
		print(f'{method:20s}: {mean[method]:6.2f}% +/- {np.sqrt(var[method]):4.2f} ({avg_time*1000:.1f} ms +/- {np.std(timing[method])*1000:.1f})')

