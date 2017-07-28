import tensorflow as tf
import numpy as np
import pandas as pd
import config
from optparse import OptionParser
from task import Task
import logging
from model_param_space import param_space_dict

def train(model_name, data_name, params_dict, logger):
	task = Task(model_name, data_name, params_dict, logger)
	task.refit()

def parse_args(parser):
	parser.add_option("-m", "--model", dest="model_name", type="string", default="best_TransE_L2")
	parser.add_option("-d", "--data", dest="data_name", type="string", default="wn18")

	options, args = parser.parse_args()
	return options, args

def main(options):
	logger = logging.getLogger()
	logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',level=logging.INFO)
	train(options.model_name, options.data_name, params_dict=param_space_dict[options.model_name], logger=logger)

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
