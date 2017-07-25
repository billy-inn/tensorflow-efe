
class Model(object):
	def __init__(self, n_entities, n_relations, hparams):
		self.n_entities = n_entities
		self.n_relations = n_relations
		self.hparams = hparams

	def add_placeholders(self):
		self.heads = tf.placeholder(tf.int32, [None], name="head_entities")
		self.tails = tf.placeholder(tf.int32, [None], name="tail_entities")
		self.relations = tf.placeholder(tf.int32, [None], name="relations")
		self.labels = tf.placeholder(tf.float32, [None], name="labels")
	
	def create_feed_dict(self, heads, tails, relations, labels=None):
		feed_dict = {
			self.heads: heads,
			self.tails: tails,
			self.relations: relations,
		}
		if labels is not None:
			feed_dict[self.labels] = labels
		return feed_dict

	def add_params(self):
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_prediction_op(self):
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_loss_op(self, pred):
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_training_op(self, loss):
		raise NotImplementedError("Each Model must re-implement this method.")

	def build(self):
		self.add_placeholders()
		self.add_params()
		self.add_prediction_op()
		self.add_loss_op()
		self.add_training_op()
