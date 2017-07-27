from model import Model
import tensorflow as tf
import datetime
import numpy as np
import config
from utils.tf_utils import *

class TransE_L2(Model):
	def __init__(self, n_entities, n_relations, hparams):
		super(TransE_L2, self).__init__(n_entities, n_relations, hparams)

		# additional params for TransE
		self.margin = hparams.margin

		# build the computation graph
		self.build()
	
	def add_params(self):
		minVal = -6/np.sqrt(self.embedding_size)
		maxVal = -minVal
		self.entity_embedding = tf.Variable(tf.random_uniform([self.n_entities, self.embedding_size], minVal, maxVal, seed=config.RANDOM_SEED), dtype=tf.float32, name="entity_embedding")
		self.relation_embedding = tf.Variable(tf.random_uniform([self.n_relations, self.embedding_size], minVal, maxVal, seed=config.RANDOM_SEED), dtype=tf.float32, name="relation_embedding")

	def add_prediction_op(self):
		self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
		self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
		self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

		self.pred = - l2_loss(self.e1 + self.r - self.e2)

	def add_loss_op(self):
		pos_size, neg_size = self.batch_size, self.batch_size * self.neg_ratio
		e1_pos, e1_neg = tf.split(self.e1, [pos_size, neg_size])
		e2_pos, e2_neg = tf.split(self.e2, [pos_size, neg_size])
		r_pos, r_neg = tf.split(self.r, [pos_size, neg_size])

		losses = tf.maximum(0.0, self.margin + l2_loss(e1_pos + r_pos - e2_pos) \
				- tf.reduce_mean(tf.reshape(l2_loss(e1_neg + r_neg - e2_neg), (self.batch_size, self.neg_ratio)), axis=-1))
		self.loss = tf.reduce_mean(losses)

	def add_training_op(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.grads_and_vars = optimizer.compute_gradients(self.loss)
		self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

	def action_before_update(self):
		self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding)

