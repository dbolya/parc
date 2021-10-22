from evaluate import Experiment
from metrics import MetricEval

from methods import PARC, kNN


# Set up the methods to use.
# To define your own method, inherit methods.TransferabilityMethod. See the methods in methods.py for more details.
my_methods = {
	'PARC f=32': PARC(n_dims=32),
	'1-NN CV'  : kNN(k=1)
}

experiment = Experiment(my_methods, name='test', append=False) # Set up an experiment with those methods named "test".
                                                               # Append=True skips evaluations that already happend. Setting it to False will overwrite.
experiment.run()                                               # Run the experiment and save results to ./results/{name}.csv

metric = MetricEval(experiment.out_file)                       # Load the experiment file we just created with the default oracle
metric.add_plasticity()                                        # Adds the "capacity to learn" heuristic defined in the paper
mean, variance, _all = metric.aggregate()                      # Compute metrics and aggregate them

# Prints {'PARC f=32': 70.27800205353863, '1-NN CV': 68.01407390300884}
print(mean)
