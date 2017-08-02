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
| TransE | TransE\_L2; TransE\_L1 |[Bordes et al. (NIPS 2013)](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) |
| NTN | | [Socher et al. (NIPS 2013)](https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf) |
| DistMult | DistMult; DistMult\_tanh; DistMult\_Logistic | [Yang et al. (ICLR 2015)](https://arxiv.org/pdf/1412.6575.pdf)
| ComplEx | Complex; Complex\_Logistic | [Trouillon et al. (ICML 2016)](https://arxiv.org/pdf/1606.06357.pdf) |

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

### Performance

<table>
<tr>
   <th>Model</th>
   <th colspan="5">WN18</th>
   <th colspan="5">FB15K</th>
</tr>
   <tr>
   <th></th>
   <th>Filtered MRR</th>
   <th>Raw MRR</th>
   <th>Hits@1 </th>
   <th>Hits@3 </th>
   <th>Hits@10 </th>
   <th>Filtered MRR</th>
   <th>Raw MRR</th>
   <th>Hits@1 </th>
   <th>Hits@3 </th>
   <th>Hits@10 </th>
   </tr>
   <tr>
   <td>TransE</td>
   <td>0.454</td>
   <td>0.338</td>
   <td>0.089</td>
   <td>0.814</td>
   <td>0.954</td>
   <td>0.407</td>
   <td>0.247</td>
   <td>0.272</td>
   <td>0.480</td>
   <td>0.657</td>    
   </tr>
   <!--<tr>
   <td>DistMult</td>
   <td>0.835</td>
   <td>0.561</td>
   <td>0.753</td>
   <td>0.913</td>
   <td>0.937</td>
   <td>0.651</td>
   <td>0.237</td>
   <td>0.544</td>
   <td>0.728</td>
   <td>0.825</td>
   </tr>
   <tr>
   <td>HolE</td>
   <td><b>0.62</b></td>
   <td><b>0.94</b></td>
   <td>0.928</td>
   <td><b>0.941</b></td>
   <td><b>0.944</b></td>
   <td>0.21</td>
   <td>0.46</td>
   <td>33.45</td>
   <td>53.63</td>
   <td>67.54</td>
   </tr>
   <tr>
   <td>ComplEx</td>
   <td>0.581</td>
   <td><b>0.94</b></td>   
   <td><b>0.937</b></td>
   <td><b>0.941</b></td>
   <td><b>0.944</b></td>
   <td><b>0.672</b></td>
   <td>0.235</td>
   <td><b>0.571</b></td>
   <td><b>0.746</b></td>
   <td><b>0.832</b></td>
   </tr>-->
</table>

### License

MIT
