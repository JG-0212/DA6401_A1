## Description ##
The repository contains
- NeuralNetwork.py : Contains the network and training related functions
- Data.py : Contains functions to load, process and visualize data
- Optimizers.py : Contains different optimizers objects
- Activations.py : Contains different activations and their derivatives
- tune.py : Hyperparameter tuning script
- train.py : Custom running script
- config.yaml : First set of hyperparameters
- config_fine.yaml : Second set of hyperparameters
- config_grid.yaml : Third set of hyperparameters

## Train and Evaluate ##
The best test performance is listed for training on full dataset with 100 epochs and scaled batch size (20x128) to account for 20 times increase in data.
However, the default hyperparameters give a very close representation of the actual performance faster and eases evaluation.
Feel free to change the default hyperparamters to the full data version and verify the reported performance

```python
python run train.py
python run train.py -e 100 -b 2560 -f 1 #for reported best test performance
```

## Additional arguments ##
-rn --runname : gives a run name to the script
-f --fraction : fraction of train dataset to train on
