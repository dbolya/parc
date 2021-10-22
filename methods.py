import numpy as np
import pickle
import pandas as pd

import constants
import datasets

import scipy.stats
import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition

# This is for Logistic so it doesn't complain that it didn't converge
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


def split_data(data:np.ndarray, percent_train:float):
	split = data.shape[0] - int(percent_train * data.shape[0])
	return data[:split], data[split:]

class TransferabilityMethod:	
	def __call__(self, 
		features:np.ndarray, probs:np.ndarray, y:np.ndarray,
		source_dataset:str, target_dataset:str, architecture:str,
		cache_path_fn) -> float:
		
		self.features = features
		self.probs = probs
		self.y = y

		self.source_dataset = source_dataset
		self.target_dataset = target_dataset
		self.architecture = architecture

		self.cache_path_fn = cache_path_fn

		# self.features = sklearn.preprocessing.StandardScaler().fit_transform(self.features)

		return self.forward()

	def forward(self) -> float:
		raise NotImplementedError




def feature_reduce(features:np.ndarray, f:int=None) -> np.ndarray:
	"""
	Use PCA to reduce the dimensionality of the features.

	If f is none, return the original features.
	If f < features.shape[0], default f to be the shape.
	"""
	if f is None:
		return features

	if f > features.shape[0]:
		f = features.shape[0]

	return sklearn.decomposition.PCA(
		n_components=f,
		svd_solver='randomized',
		random_state=1919,
		iterated_power=1).fit_transform(features)




class LEEP(TransferabilityMethod):
	"""
	LEEP: https://arxiv.org/abs/2002.12462
	
	src ('probs', 'features') denotes what to use for leep.
	
	normalization ('l1', 'softmax'). The normalization strategy to get everything to sum to 1.
	"""

	def __init__(self, n_dims:int=None, src:str='probs', normalization:str=None, use_sigmoid:bool=False):
		self.n_dims = n_dims
		self.src = src
		self.normalization = normalization
		self.use_sigmoid = use_sigmoid

	def forward(self) -> float:
		theta = getattr(self, self.src)
		y = self.y
		
		n = theta.shape[0]
		n_y = constants.num_classes[self.target_dataset]

		# n             : Number of target data images
		# n_z           : Number of source classes
		# n_y           : Number of target classes
		# theta [n, n_z]: The source task probabilities on the target images
		# y     [n]     : The target dataset label indices {0, ..., n_y-1} for each target image

		unnorm_prob_joint    = np.eye(n_y)[y, :].T @ theta                       # P(y, z): [n_y, n_z]
		unnorm_prob_marginal = theta.sum(axis=0)                                 # P(z)   : [n_z]
		prob_conditional     = unnorm_prob_joint / unnorm_prob_marginal[None, :] # P(y|z) : [n_y, n_z]

		leep = np.log((prob_conditional[y] * theta).sum(axis=-1)).sum() / n      # Eq. 2

		return leep


class NegativeCrossEntropy(TransferabilityMethod):
	""" NCE: https://arxiv.org/pdf/1908.08142.pdf """

	def forward(self, eps=1e-5) -> float:
		z = self.probs.argmax(axis=-1)

		n = self.y.shape[0]
		n_y = constants.num_classes[self.target_dataset]
		n_z = constants.num_classes[self.source_dataset]

		prob_joint    = (np.eye(n_y)[self.y, :].T @ np.eye(n_z)[z, :]) / n + eps
		prob_marginal = np.eye(n_z)[z, :].sum(axis=0) / n + eps

		NCE = (prob_joint * np.log(prob_joint / prob_marginal[None, :])).sum()

		return NCE

		
class HScore(TransferabilityMethod):
	""" HScore from https://ieeexplore.ieee.org/document/8803726 """

	def __init__(self, n_dims:int=None, use_published_implementation:bool=False):
		self.use_published_implementation = use_published_implementation
		self.n_dims = n_dims

	def getCov(self, X):
		X_mean= X - np.mean(X,axis=0,keepdims=True)
		cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
		return cov

	def getHscore(self, f,Z):
		Covf = self.getCov(f)
		g = np.zeros_like(f)
		for z in range(constants.num_classes[self.target_dataset]):
			idx = (Z == z)
			if idx.any():
				Ef_z=np.mean(f[idx, :], axis=0)
				g[idx]=Ef_z
		
		Covg=self.getCov(g)
		score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg))

		return score

	def get_hscore_fast(self, eps=1e-8):
		# The original implementation of HScore isn't properly vectorized, so do that here
		cov_f = self.getCov(self.features)
		n_y = constants.num_classes[self.target_dataset]

		# Vectorize the inner loop over each class
		one_hot_class = np.eye(n_y)[self.y, :]   # [#probe, #classes]
		class_counts = one_hot_class.sum(axis=0) # [#classes]

		# Compute the mean feature per class
		mean_features = (one_hot_class.T @ self.features) / (class_counts[:, None] + eps) # [#classes, #features]

		# Redistribute that into the original features' locations
		g = one_hot_class @ mean_features # [#probe, #features]
		cov_g = self.getCov(g)
		
		score = np.trace(np.linalg.pinv(cov_f, rcond=1e-15) @ cov_g)

		return score
		

	def forward(self):
		self.features = feature_reduce(self.features, self.n_dims)

		scaler = sklearn.preprocessing.StandardScaler()
		self.features = scaler.fit_transform(self.features)

		if self.use_published_implementation:
			return self.getHscore(self.features, self.y)
		else:
			return self.get_hscore_fast()



class kNN(TransferabilityMethod):
	"""
	k Nearest Neighbors with hold-one-out cross-validation.

	Metric can be one of (euclidean, cosine, cityblock)

	This method supports VOC2007.
	"""

	def __init__(self, k:int=1, metric:str='l2', n_dims:int=None):
		self.k = k
		self.metric = metric
		self.n_dims = n_dims
	
	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)

		dist = sklearn.metrics.pairwise_distances(self.features, metric=self.metric)
		idx = np.argsort(dist, axis=-1)

		# After sorting, the first index will always be the same element (distance = 0), so choose the k after
		idx = idx[:, 1:self.k+1]

		votes = self.y[idx]
		preds, counts = scipy.stats.mode(votes, axis=1)

		n_data = self.features.shape[0]

		preds = preds.reshape(n_data, -1)
		counts = counts.reshape(n_data, -1)
		votes = votes.reshape(n_data, -1)

		preds = np.where(counts == 1, votes, preds)

		return 100*(preds == self.y.reshape(n_data, -1)).mean()
		# return -np.abs(preds - self.y).sum(axis=-1).mean() # For object detection

class SplitkNN(TransferabilityMethod):
	""" k Nearest Neighbors using a train-val split using sklearn. Only supports l2 distance. """

	def __init__(self, percent_train:float=0.5, k:int=1, n_dims:int=None):
		self.percent_train = percent_train
		self.k = k
		self.n_dims = n_dims

	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)

		train_x, test_x = split_data(self.features, self.percent_train)
		train_y, test_y = split_data(self.y       , self.percent_train)

		nn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.k).fit(train_x, train_y)
		return 100*(nn.predict(test_x) == test_y).mean()


class SplitLogistic(TransferabilityMethod):
	""" Logistic classifier using a train-val split using sklearn. """

	def __init__(self, percent_train:float=0.5, n_dims:int=None):
		self.percent_train = percent_train
		self.n_dims = n_dims
		
	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)
		
		train_x, test_x = split_data(self.features, self.percent_train)
		train_y, test_y = split_data(self.y       , self.percent_train)

		logistic = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=20, tol=1e-1).fit(train_x, train_y)
		return 100*(logistic.predict(test_x) == test_y).mean()


class RSA(TransferabilityMethod):
	"""
	Computes the RSA similarity metric proposed in https://arxiv.org/abs/1904.11740. 
	
	Note that this requires the probes to be fully extracted before running.

	This method supports VOC2007.
	"""
	def __init__(self, reference_architecture:str=None, n_dims:int=None):
		self.reference_architecture = reference_architecture
		self.n_dims = n_dims

	def forward(self):
		self.features = feature_reduce(self.features, self.n_dims)
		
		reference_architecture = self.reference_architecture if self.reference_architecture is not None else self.architecture

		with open(self.cache_path_fn(reference_architecture, self.target_dataset, self.target_dataset), 'rb') as f:
			reference_params = pickle.load(f)
		
		reference_features = reference_params['features']
		reference_features = feature_reduce(reference_features, self.n_dims)
		
		return self.get_rsa_correlation(self.features, reference_features)
	
	def get_rsa_correlation(self, feats1:np.ndarray, feats2:np.ndarray) -> float:
		scaler = sklearn.preprocessing.StandardScaler()
		
		feats1 = scaler.fit_transform(feats1)
		feats2 = scaler.fit_transform(feats2)

		rdm1 = 1 - np.corrcoef(feats1)
		rdm2 = 1 - np.corrcoef(feats2)

		lt_rdm1 = self.get_lowertri(rdm1)
		lt_rdm2 = self.get_lowertri(rdm2)

		return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100
	
	def get_lowertri(self, rdm):
		num_conditions = rdm.shape[0]
		return rdm[np.triu_indices(num_conditions,1)]




class PARC(TransferabilityMethod):
	"""
	Computes PARC, a variation of RSA that uses target labels instead of target features to cut down on training time.
	This was presented in this paper.
	
	This method supports VOC2007.
	"""

	def __init__(self, n_dims:int=None, fmt:str=''):
		self.n_dims = n_dims
		self.fmt = fmt

	def forward(self):
		self.features = feature_reduce(self.features, self.n_dims)
		
		num_classes = constants.num_classes[self.target_dataset]
		labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

		return self.get_parc_correlation(self.features, labels)

	def get_parc_correlation(self, feats1, labels2):
		scaler = sklearn.preprocessing.StandardScaler()

		feats1  = scaler.fit_transform(feats1)

		rdm1 = 1 - np.corrcoef(feats1)
		rdm2 = 1 - np.corrcoef(labels2)
		
		lt_rdm1 = self.get_lowertri(rdm1)
		lt_rdm2 = self.get_lowertri(rdm2)
		
		return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

	def get_lowertri(self, rdm):
		num_conditions = rdm.shape[0]
		return rdm[np.triu_indices(num_conditions,1)]




class DDS(TransferabilityMethod):
	"""
	DDS from https://github.com/cvai-repo/duality-diagram-similarity/
	
	This method supports VOC2007.
	"""

	def __init__(self, reference_architecture:str=None, n_dims:int=None):
		self.reference_architecture = reference_architecture
		self.n_dims = n_dims

	def forward(self):
		self.features = feature_reduce(self.features, self.n_dims)
		
		reference_architecture = self.reference_architecture if self.reference_architecture is not None else self.architecture

		with open(self.cache_path_fn(reference_architecture, self.target_dataset, self.target_dataset), 'rb') as f:
			reference_params = pickle.load(f)
		
		reference_features = reference_params['features']
		reference_features = feature_reduce(reference_features, self.n_dims)
		
		return self.get_similarity_from_rdms(self.features, reference_features)

	
	def rdm(self, activations_value,dist):
		"""
		Parameters
		----------
		activations_value : numpy matrix with dimensions n x p 
			task 1 features (n = number of images, p = feature dimensions) 
		dist : string
			distance function to compute dissimilarity matrix
		Returns
		-------
		RDM : numpy matrix with dimensions n x n 
			dissimilarity matrices
		"""
		if dist == 'pearson':
			RDM = 1 - np.corrcoef(activations_value)
		elif dist == 'cosine':
			RDM = 1 - sklearn.metrics.pairwise.cosine_similarity(activations_value)
		return RDM


	def get_similarity_from_rdms(self, x,y,debiased=True,centered=True):
		"""
		Parameters
		----------
		x : numpy matrix with dimensions n x p 
			task 1 features (n = number of images, p = feature dimensions) 
		y : numpy matrix with dimensions n x p
			task 1 features (n = number of images, p = feature dimensions) 
		dist : string
			distance function to compute dissimilarity matrices
		feature_norm : string
			feature normalization type
		debiased : bool, optional
			set True to perform unbiased centering 
		centered : bool, optional
			set True to perform unbiased centering 
		Returns
		-------
		DDS: float
			DDS between task1 and task2 
		"""
		x = sklearn.preprocessing.StandardScaler().fit_transform(x)
		y = sklearn.preprocessing.StandardScaler().fit_transform(y)
		
		return self.cka(self.rdm(x, 'cosine'), self.rdm(y, 'cosine'), debiased=debiased,centered=centered) * 100

	def center_gram(self, gram, unbiased=False):
		"""
		Center a symmetric Gram matrix.
		
		This is equvialent to centering the (possibly infinite-dimensional) features
		induced by the kernel before computing the Gram matrix.
		
		Args:
			gram: A num_examples x num_examples symmetric matrix.
			unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
			estimate of HSIC. Note that this estimator may be negative.
		Returns:
			A symmetric matrix with centered columns and rows.
		
		P.S. Function from Kornblith et al., ICML 2019
		"""
		if not np.allclose(gram, gram.T):
			raise ValueError('Input must be a symmetric matrix.')
		gram = gram.copy()

		if unbiased:
			# This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
			# L. (2014). Partial distance correlation with methods for dissimilarities.
			# The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
			# stable than the alternative from Song et al. (2007).
			n = gram.shape[0]
			np.fill_diagonal(gram, 0)
			means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
			means -= np.sum(means) / (2 * (n - 1))
			gram -= means[:, None]
			gram -= means[None, :]
			np.fill_diagonal(gram, 0)
		else:
			means = np.mean(gram, 0, dtype=np.float64)
			means -= np.mean(means) / 2
			gram -= means[:, None]
			gram -= means[None, :]

		return gram


	def cka(self, gram_x, gram_y, debiased=False,centered=True):
		"""
		Compute CKA.
		Args:
			gram_x: A num_examples x num_examples Gram matrix.
			gram_y: A num_examples x num_examples Gram matrix.
			debiased: Use unbiased estimator of HSIC. CKA may still be biased.
		Returns:
			The value of CKA between X and Y.
			P.S. Function from Kornblith et al., ICML 2019
		"""
		if centered:
			gram_x = self.center_gram(gram_x, unbiased=debiased)
			gram_y = self.center_gram(gram_y, unbiased=debiased)

		# Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
		# n*(n-3) (unbiased variant), but this cancels for CKA.
		scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

		normalization_x = np.linalg.norm(gram_x)
		normalization_y = np.linalg.norm(gram_y)
		
		return scaled_hsic / (normalization_x * normalization_y)



class LearnedHeuristic():

	def __init__(self, cache_file:str='./cache/learned_heuristic.pkl'):
		self.cache_file = cache_file

	def predict(self, x:list) -> float:
		return sum([a * x_i for a, x_i in zip(self.coeffs, x)])

	def make_feature(self, arch:str, source:str, target:str) -> list:
		feats = [
			constants.num_classes[source],
			constants.num_classes[target],
			constants.dataset_images[source],
			constants.dataset_images[target],
			constants.model_layers[arch]
		]

		return feats + [np.log(x) for x in feats]

	def fit(self, oracle_path:str, percent_train:float=0.5):
		oracle = pd.read_csv(oracle_path)

		x = []
		y = []

		for idx, row in oracle.iterrows():
			arch   = row['Architecture']
			source = row['Source Dataset']
			target = row['Target Dataset']

			x.append(self.make_feature(arch, source, target))
			y.append(row['Oracle'])
		
		x = np.array(x)
		y = np.array(y)

		regr = sklearn.linear_model.LinearRegression()
		regr.fit(x, y)
		
		self.coeffs = list(regr.coef_)

		with open(self.cache_file, 'wb') as f:
			pickle.dump(self.coeffs, f)
	
	def load(self):
		with open(self.cache_file, 'rb') as f:
			self.coeffs = pickle.load(f)






methods = {
	# 'LEEP': LEEP(),
	# 'NCE': NegativeCrossEntropy(),
	# 'HScore': HScore(use_published_implementation=True),
	
	# '1-NN CV': kNN(k=1),
	# '5-NN CV': kNN(k=5),

	# '1-NN': SplitkNN(k=1),
	# '5-NN': SplitkNN(k=5),

	# 'Logistic': SplitLogistic(),

	# 'DDS Full': DDS(),
	# 'DDS Resnet-50': DDS(reference_architecture='resnet50'),
	# 'DDS Resnet-18': DDS(reference_architecture='resnet18'),
	# 'DDS Alexnet': DDS(reference_architecture='alexnet'),
	# 'DDS GoogLeNet': DDS(reference_architecture='googlenet'),

	# 'RSA Full': RSA(),
	# 'RSA Resnet-50': RSA(reference_architecture='resnet50'),
	# 'RSA Resnet-18': RSA(reference_architecture='resnet18'),
	# 'RSA Alexnet': RSA(reference_architecture='alexnet'),
	# 'RSA GoogLeNet': RSA(reference_architecture='googlenet'),

	# 'PARC': PARC(),
	

	
	# 'HScore  16' : HScore(use_published_implementation=True, n_dims= 16),
	# 'HScore  32' : HScore(use_published_implementation=True, n_dims= 32),
	# 'HScore  64' : HScore(use_published_implementation=True, n_dims= 64),
	# 'HScore 128' : HScore(use_published_implementation=True, n_dims=128),
	# 'HScore 256' : HScore(use_published_implementation=True, n_dims=256),
	
	# '1-NN CV  16': kNN(k=1, n_dims= 16),
	# '1-NN CV  32': kNN(k=1, n_dims= 32),
	# '1-NN CV  64': kNN(k=1, n_dims= 64),
	# '1-NN CV 128': kNN(k=1, n_dims=128),
	# '1-NN CV 256': kNN(k=1, n_dims=256),

	# 'Logistic  16': SplitLogistic(n_dims= 16),
	# 'Logistic  32': SplitLogistic(n_dims= 32),
	# 'Logistic  64': SplitLogistic(n_dims= 64),
	# 'Logistic 128': SplitLogistic(n_dims=128),
	# 'Logistic 256': SplitLogistic(n_dims=256),

	# 'RSA Resnet-50  16': RSA(reference_architecture='resnet50', n_dims= 16),
	# 'RSA Resnet-50  32': RSA(reference_architecture='resnet50', n_dims= 32),
	# 'RSA Resnet-50  64': RSA(reference_architecture='resnet50', n_dims= 64),
	# 'RSA Resnet-50 128': RSA(reference_architecture='resnet50', n_dims=128),
	# 'RSA Resnet-50 256': RSA(reference_architecture='resnet50', n_dims=256),
	
	# 'DDS Resnet-50  16': DDS(reference_architecture='resnet50', n_dims= 16),
	# 'DDS Resnet-50  32': DDS(reference_architecture='resnet50', n_dims= 32),
	# 'DDS Resnet-50  64': DDS(reference_architecture='resnet50', n_dims= 64),
	# 'DDS Resnet-50 128': DDS(reference_architecture='resnet50', n_dims=128),
	# 'DDS Resnet-50 256': DDS(reference_architecture='resnet50', n_dims=256),

	# 'PARC  16': PARC(n_dims= 16),
	# 'PARC  32': PARC(n_dims= 32),
	# 'PARC  64': PARC(n_dims= 64),
	# 'PARC 128': PARC(n_dims=128),
	# 'PARC 256': PARC(n_dims=256),


	# 'LEEP': LEEP(),
	# 'NCE': NegativeCrossEntropy(),
	# 'HScore': HScore(use_published_implementation=True),
	'1-NN CV': kNN(k=1),
	# '5-NN CV': kNN(k=5),
	# 'Logistic': SplitLogistic(),
	# 'RSA Resnet-50': RSA(reference_architecture='resnet50'),
	# 'DDS Resnet-50': DDS(reference_architecture='resnet50'),
	'PARC': PARC(),

	# 'HScore  32' : HScore(use_published_implementation=True, n_dims= 32),
	# 'Logistic 256': SplitLogistic(n_dims=256),
	
	'1-NN CV 256': kNN(k=1, n_dims=256),
	'PARC  32': PARC(n_dims= 32),
	# 'RSA Resnet-50  64': RSA(reference_architecture='resnet50', n_dims= 64),
	# 'DDS Resnet-50  64': DDS(reference_architecture='resnet50', n_dims= 64),
}


if __name__ == '__main__':
	def cache_path(architecture:str, source_dataset:str, target_dataset:str):
		return f'./cache/probes/fixed_budget_500/{architecture}_{source_dataset}_{target_dataset}_{1}.pkl'

	with open(cache_path('resnet50', 'panoptic-rcnn_fpn_139514544', 'voc2007'), 'rb') as f:
		params = pickle.load(f)
		params['cache_path_fn'] = cache_path
	
	print(' -- Good transfer -- ')

	for name, method in methods.items():
		import time

		a = time.time()
		print(f'{name:20s}: {method(**params):6.2f}')
		# print(time.time() - a)

	
	with open(cache_path('resnet50', 'faster-rcnn_fpn_137849458', 'voc2007'), 'rb') as f:
		params = pickle.load(f)
		params['cache_path_fn'] = cache_path
	
	print()
	print(' -- Bad transfer -- ')

	for name, method in methods.items():
		print(f'{name:20s}: {method(**params):6.2f}')
