# PARC for Scalable Diverse Model Selection

Here, we present a set of benchmarks for Scalable Diverse Model Selection.
We also include our method, PARC, as a good baseline on this benchmark.

This is the code for our NeurIPS 2021 paper available [here](coming_soon) (coming soon).

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/1Fl6PBpdQgg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Installing the Benchmark
To install, first clone this repo:
```
git clone https://github.com/dbolya/parc.git
cd parc
```

Make sure your python version is at least 3.7.
Then install the requirements:
```
pip install -r requirements.txt
```

Finally, download the cached probe sets from here:   [500 Image Probe Sets](https://gtvault-my.sharepoint.com/:u:/g/personal/jhoffman68_gatech_edu/ET7lrR7bVLpPqGSvjud6pvcBFS4WSbDEjCozVQw-x3O0fw?e=4mQhfV)

Then extract the probes into `./cache/` (or symlink it there):
```
unzip probes.zip -d ./cache/
```

Verify that the probe set exists
```
ls ./cache/probes/fixed_budget_500 | head
```
You should see a couple of probe sets `pkl` files.

And you're done! If you want to create your own probe sets see the `Advanced` section below.



## Evaluation
See `demo.py` for an example of how to perform evaluation:
```py
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

# Prints {'PARC f=32': 70.27800205353863, '1-NN CV': 68.01407390300884}. Same as Table 4 in the paper.
print(mean)
```



If you wish to use the extended set of transfers (using crowd-sourced models), pass `model_bank='all'` to the experiment and pass `oracle_path='./oracles/all.csv'` when creating the metric evaluation object.



## PARC
If you wish to use PARC to recommend models for you, PARC is defined in `methods.py`. We don't have a well supported way for you to pass arbitrary data in yet, but as long as you pass in everything required in `TransferabilityMethod`, you should be fine.


## Advanced
If you want the trained models, they are available here:
[All Trained Models](https://gtvault-my.sharepoint.com/:u:/g/personal/jhoffman68_gatech_edu/EQLLyM-kQsBNqYYWjW-l6NMBSCCuOouP8tStz5vOqutJYg?e=djJ4Ni). Note that this only includes the models we trained from scratch, not the crowd sourced models.

If you want to create the probe sets yourself, put / symlink the datasets as `./data/{dataset}/`. Then put the models above in `./models/`. This is not necessary if you use the pre-extracted probe sets instead.


## Citation
If you used PARC, this benchmark, or this code your work, please cite:
```
@inproceedings{parc-neurips2021,
  author    = {Daniel Bolya and Rohit Mittapalli and Judy Hoffman},
  title     = {Scalable Diverse Model Selection for Accessible Transfer Learning},
  booktitle = {NeurIPS},
  year      = {2021},
}
```
