
## Model Ensemble

To achieve robust predictions of 

<div align="center"> <img src="../figures/ensemble.png" width="600"/> </div>

### How to Run Model Ensemble Independently

```python
## model ensemble
n_best = 1
builder = AlgoEnsembleBuilder(history, data_src_cfg)
builder.set_ensemble_method(AlgoEnsembleBestN(n_best=n_best))
ensemble = builder.get_ensemble()
pred = ensemble()
print("ensemble picked the following best {0:d}:".format(n_best))
for algo in ensemble.get_algo_ensemble():
    print(algo[AlgoEnsembleKeys.ID])
```

### Essential Component for General Algorithm/Mdoel Ensemble

