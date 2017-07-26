import numpy as np
from hyperopt import hp

# required params: 
# - embedding_size
# - lr
# - batch_size
# - max_iter
# - neg_ratio
# - contiguous_sampling
# - valid_every

param_space_TransE_L2 = {
	"embedding_size": 50,
	"margin": 1.0,
	"lr": 0.01,
	"batch_size": 100,
	"max_iter": 1000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 100,
}

param_space_dict = {
	"TransE_L2": param_space_TransE_L2,
}

int_params = [
	"embedding_size", "batch_size", "max_iter", "neg_ratio",
]

class ModelParamSpace:
	def __init__(self, learner_name):
		s = "Wrong learner name!"
		assert learner_name in param_space_dict, s
		self.learner_name = learner_name
	
	def _build_space(self):
		return param_space_dict[self.learner_name]

	def _convert_into_param(self, param_dict):
		if isinstance(param_dict, dict):
			for k,v in param_dict.items():
				if k in int_params:
					param_dict[k] = int(v)
				elif isinstance(v, list) or isinstance(v, tuple):
					for i in range(len(v)):
						self._convert_into_param(v[i])
				elif isinstance(v, dict):
					self._convert_into_param(v)
		return param_dict
