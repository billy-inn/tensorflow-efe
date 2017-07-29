# TensorFlow-EFE

A collection of Tensorflow implementations of embeddings for entities.

### Requirements

- Python 3
- Tensorflow >= 1.0
- Hyperopt

### Models

The generic abstract model is defined in [model.py](https://github.com/billy-inn/tensorflow-efe/blob/master/model.py). 
All specific models are implemented in [efe.py](https://github.com/billy-inn/tensorflow-efe/blob/master/efe.py)

| Model | Implementations | Reference |
| :---- | :-------------- | :-------- |
| TransE | L2; L1 |[Bordes et al. (NIPS 2013)](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)

### Hyperparameters

#### Set hyperparameters

Add hyperparameters dict and its identifier in [model_param_space.py](https://github.com/billy-inn/tensorflow-efe/blob/master/model_param_space.py).

#### Search optimal hyperparameters

`python task.py -m [model_name] -d [data_name] -e [max_evals] -c [cv_runs]`

*model\_name* is the identifier defined in the [model_param_space.py](https://github.com/billy-inn/tensorflow-efe/blob/master/model_param_space.py). *data\_name* is either **wn18** or **fb15k**. *max\_evals* is the maximum runs to search the hyperparameters, default: 100. *cv\_runs* is the number of runs for the cross validation, default: 3. 

The search process and result are stored in `log` folder.

### Evaluation

`python train.py -m [model_name] -d [data_name]`

Train on the given hyperparameter setting and give the result for the test set.

### License

MIT
