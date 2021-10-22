from evaluate import Experiment
from metrics import MetricEval

from methods import PARC, kNN


my_methods = {
	'PARC f=32'    : PARC(n_dims=32),
	'kNN k=1,f=128': kNN(k=1, n_dims=128)
}

out_path = Experiment(my_methods, name='test').run()
result = MetricEval(out_path).aggregate(my_methods)

print(result)
