import tensorflow as tf
from optparse import OptionParser
import numpy as np
import os
import config
from utils.data_utils import load_dict_from_txt

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

def get_complex_scores(model_name, output_path):
	checkpoint_file = os.path.join(config.CHECKPOINT_PATH, model_name)
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		heads = graph.get_operation_by_name("head_entities").outputs[0]
		tails = graph.get_operation_by_name("tail_entities").outputs[0]
		relations = graph.get_operation_by_name("relations").outputs[0]
		pred = graph.get_operation_by_name("pred").outputs[0]

		entity2id = load_dict_from_txt("../NeuralRE/data/entity2id.txt")
		relation2id = load_dict_from_txt("../NeuralRE/data/relation2id.txt")
		id2e = {entity2id[x]:x for x in entity2id.keys()}
		id2r = {relation2id[x]:x for x in relation2id.keys()}
		e2id = load_dict_from_txt(config.FB1M_E2ID)
		r2id = load_dict_from_txt(config.FB1M_R2ID)
		r = []
		for i in range(1,55):
			r.append(r2id[id2r[i]])
		res = sess.run(pred, feed_dict={heads: [e2id[id2r[8140]]]*54, tails: [e2id[id2r[13196]]]*54, relations: r})

		with open(output_path, "w") as f:
			for x, y in enumerate(res[0]):
				f.write("%d %f\n" % (x, y))

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	#get_complex_embeddings(options.model_name, options.output_path)
	get_complex_scores(options.model_name, options.output_path)
