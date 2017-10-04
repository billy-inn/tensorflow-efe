import tensorflow as tf
from optparse import OptionParser
import numpy as np
import os
import config

def parse_args(parser):
	parser.add_option("-m", "--model", dest="model_name", type="string")
	parser.add_option("-o", "--output", dest="output_path", type="string")
	options, args = parser.parse_args()
	return options, args

def get_embeddings(model_name, output_path):
	checkpoint_file = os.path.join(config.CHECKPOINT_PATH, model_name)
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		entity = graph.get_tensor_by_name("entity_embedding:0")
		relation = graph.get_tensor_by_name("relation_embedding:0")
		e, r = sess.run([entity, relation])
		np.save(os.path.join(output_path, "entity.npy"), e)
		np.save(os.path.join(output_path, "relation.npy"), r)

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	get_embeddings(options.model_name, options.output_path)
