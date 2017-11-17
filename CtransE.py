from model import Model
import tensorflow as tf
import numpy as np
import datetime
import config
from utils.tf_utils import l2_loss, l1_loss

class CTransE(Model):
    def __init__(self, n_entities, n_relations, hparams, sub_mat, obj_mat):
        super(CTransE, self).__init__(n_entities, n_relations, hparams)
        self.margin = hparams.margin
        # self.sub_mat = tf.SparseTensorValue(
        #    indices=np.array([sub_mat.row, sub_mat.col]).T,
        #    values=sub_mat.data,
        #    dense_shape=sub_mat.shape)
        # self.obj_mat = tf.SparseTensorValue(
        #    indices=np.array([obj_mat.row, obj_mat.col]).T,
        #    values=obj_mat.data,
        #    dense_shape=obj_mat.shape)
        sub_idx = sorted(list(zip(sub_mat.row, sub_mat.col)))
        self.sub_mat = tf.sparse_to_dense(
            sparse_indices=np.array([[x, y] for x, y in sub_idx]),
            sparse_values=sub_mat.data,
            output_shape=sub_mat.shape)
        obj_idx = sorted(list(zip(obj_mat.row, obj_mat.col)))
        self.obj_mat = tf.sparse_to_dense(
            sparse_indices=np.array([[x, y] for x, y in obj_idx]),
            sparse_values=obj_mat.data,
            output_shape=obj_mat.shape)
        self.build()

    def add_params(self):
        minVal = -6 / np.sqrt(self.embedding_size)
        maxVal = -minVal
        self.entity_embedding = tf.Variable(tf.random_uniform(
            [self.n_entities, self.embedding_size], minVal, maxVal,
            seed=config.RANDOM_SEED), name="entity_embedding")
        self.relation_embedding = tf.Variable(tf.random_uniform(
            [self.n_relations, self.embedding_size], minVal, maxVal,
            seed=config.RANDOM_SEED), name="relation_embedding")

        normalized_entity_embedding = tf.nn.l2_normalize(self.entity_embedding, -1)
        self.normalization = self.entity_embedding.assign(normalized_entity_embedding)

    def train_on_batch(self, sess, input_batch):
        feed = self.create_feed_dict(**input_batch)
        sess.run(self.normalization)
        _, step, loss = sess.run([self.train_op, self.global_step, self.loss],
                feed_dict=feed)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

class CTransE_L2(CTransE):
    def add_prediction_op(self):
        self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
        self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)
        self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

        self.pred = tf.negative(l2_loss(self.e1 + self.r - self.e2), name="pred")

    def add_loss_op(self):
        pos_size, neg_size = self.batch_size, self.batch_size * self.neg_ratio * 2
        rel_pos, rel_neg = tf.split(self.relations, [pos_size, neg_size])
        e1_pos, e1_neg = tf.split(self.e1, [pos_size, neg_size])
        e2_pos, e2_neg = tf.split(self.e2, [pos_size, neg_size])
        score_pos, score_neg = tf.split(self.pred, [pos_size, neg_size])

        c1 = tf.to_float(tf.nn.embedding_lookup(self.sub_mat, rel_pos))
        c2 = tf.to_float(tf.nn.embedding_lookup(self.obj_mat, rel_pos))
        centroid1 = tf.div(tf.matmul(c1, self.entity_embedding),
                tf.reduce_sum(c1, 1, keep_dims=True))
        centroid2 = tf.div(tf.matmul(c2, self.entity_embedding),
                tf.reduce_sum(c2, 1, keep_dims=True))

        losses = tf.maximum(0.0, self.margin - score_pos + tf.reduce_mean(
            tf.reshape(score_neg, (self.batch_size, self.neg_ratio * 2)), -1))
        regularizer = l2_loss(e1_pos - centroid1) + l2_loss(e2_pos - centroid2)
        self.loss = tf.add(tf.reduce_mean(losses),
                tf.reduce_mean(0.1 * regularizer), name="loss")
