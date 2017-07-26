import tensorflow as tf
import numpy as np
import pandas as pd
import config
from efe import TransE_L2
from utils.data_utils import load_dict_from_txt
from optparse import OptionParser
from model_param_space import param_space_dict

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def train(model_name, data_name, params_dict):
	hparams = AttrDict(params_dict)

	if data_name == "wn18":
		train_triples = pd.read_csv(config.WN18_TRAIN, names=["e1", "r", "e2"]).as_matrix()
		valid_triples = pd.read_csv(config.WN18_VALID, names=["e1", "r", "e2"]).as_matrix()
		test_triples = pd.read_csv(config.WN18_TEST, names=["e1", "r", "e2"]).as_matrix()
		e2id = load_dict_from_txt(config.WN18_E2ID)
		r2id = load_dict_from_txt(config.WN18_R2ID)
	else:
		train_triples = pd.read_csv(config.FB15K_TRAIN, names=["e1", "r", "e2"]).as_matrix()
		valid_triples = pd.read_csv(config.FB15K_VALID, names=["e1", "r", "e2"]).as_matrix()
		test_triples = pd.read_csv(config.FB15K_TEST, names=["e1", "r", "e2"]).as_matrix()
		e2id = load_dict_from_txt(config.FB15K_E2ID)
		r2id = load_dict_from_txt(config.FB15K_R2ID)
	
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			model = TransE_L2(len(e2id), len(r2id), hparams)

			sess.run(tf.global_variables_initializer())

			model.fit(sess, train_triples, valid_triples)

		sess.close()
	tf.reset_default_graph()
	return 0.0

def parse_args(parser):
	parser.add_option("-m", "--model", dest="model_name", type="string", default="TransE_L2")
	parser.add_option("-d", "--data", dest="data_name", type="string", default="wn18")

	options, args = parser.parse_args()
	return options, args

def main(options):
	train(options.model_name, options.data_name, params_dict=param_space_dict[options.model_name])

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
