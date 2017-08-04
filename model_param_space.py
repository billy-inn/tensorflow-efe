import numpy as np
from hyperopt import hp

# required params: 
# - embedding_size
# - lr
# - batch_size
# - max_iter
# - neg_ratio
# - contiguous_sampling
# - valid_every: set it to 0 to enable early stopping

param_space_TransE = {
	"embedding_size": hp.quniform("embedding_size", 50, 200, 10),
	"margin": hp.quniform("margin", 0.5, 5, 0.5),
	"lr": hp.qloguniform("lr", np.log(1e-3), np.log(1e-1), 1e-3),
	"batch_size": 5000,
	"max_iter": 100000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 1000,
}

param_space_best_TransE_L2_wn18 = {
	"embedding_size": 200,
	"margin": 0.5,
	"lr": 0.001,
	"batch_size": 2000,
	"max_iter": 2000,
	"neg_ratio": 10,
	"contiguous_sampling": False,
	"valid_every": 0,
}

param_space_best_TransE_L1_fb15k = {
	"embedding_size": 190,
	"margin": 3.5,
	"lr": 0.001,
	"batch_size": 5000,
	"max_iter": 15000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 0,
}

param_space_DistMult = {
	"embedding_size": hp.quniform("embedding_size", 50, 200, 10),
	"l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-4), np.log(1e-3), 1e-4),
	"lr": hp.qloguniform("lr", np.log(1e-3), np.log(1e-1), 1e-3),
	"batch_size": 5000,
	"max_iter": 100000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 5000,
}

param_space_best_DistMult = {
	"embedding_size": 200,
	"l2_reg_lambda": 0.003,
	"lr": 0.1,
	"batch_size": 2000,
	"max_iter": 2000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 0,
}

param_space_best_NTN = {
	"embedding_size": hp.quniform("embedding_size", 50, 200, 10),
	"k": 2,
	"l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-4), np.log(1e-3), 1e-4),
	"lr": hp.qloguniform("lr", np.log(1e-3), np.log(1e-1), 1e-3),
	"batch_size": 2000,
	"max_iter": 100000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 5000,
}

param_space_best_NTN = {
	"embedding_size": 50,
	"k": 2,
	"l2_reg_lambda": 0.003,
	"lr": 0.01,
	"batch_size": 2000,
	"max_iter": 2000,
	"neg_ratio": 1,
	"contiguous_sampling": False,
	"valid_every": 100,
}

param_space_dict = {
	"TransE_L2": param_space_TransE,
	"TransE_L1": param_space_TransE,
	"best_TransE_L2_wn18": param_space_best_TransE_L2_wn18,
	"best_TransE_L1_fb15k": param_space_best_TransE_L1_fb15k,
	"DistMult": param_space_DistMult,
	"DistMult_tanh": param_space_DistMult,
	"best_DistMult": param_space_best_DistMult,
	"best_DistMult_tanh": param_space_best_DistMult,
	"NTN": param_space_NTN,
	"best_NTN": param_space_best_NTN,
}

int_params = [
	"embedding_size", "batch_size", "max_iter", "neg_ratio", "valid_every", "k",
]

class ModelParamSpace:
	def __init__(self, learner_name):
		s = "Invalid model name! (Check model_param_space.py)"
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
