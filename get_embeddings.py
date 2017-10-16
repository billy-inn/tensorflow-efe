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

def get_distmult_embeddings(model_name, output_path):
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

def get_complex_embeddings(model_name, output_path):
	checkpoint_file = os.path.join(config.CHECKPOINT_PATH, model_name)
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		entity1 = graph.get_tensor_by_name("entity_embedding1:0")
		entity2 = graph.get_tensor_by_name("entity_embedding2:0")
		relation1 = graph.get_tensor_by_name("relation_embedding1:0")
		relation2 = graph.get_tensor_by_name("relation_embedding2:0")
		e1, e2, r1, r2 = sess.run([entity1, entity2, relation1, relation2])
		np.save(os.path.join(output_path, "entity1.npy"), e1)
		np.save(os.path.join(output_path, "entity2.npy"), e2)
		np.save(os.path.join(output_path, "relation1.npy"), r1)
		np.save(os.path.join(output_path, "relation2.npy"), r2)

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	get_complex_embeddings(options.model_name, options.output_path)
