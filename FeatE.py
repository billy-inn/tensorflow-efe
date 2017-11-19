from model import Model
import tensorflow as tf
import numpy as np
import datetime
import config

class FeatE(Model):
    def add_hidden_layer(self, x, idx):
        dim = self.feat_size if idx == 0 else self.hidden_size
        with tf.variable_scope("hidden_%d" % idx):
            W = tf.get_variable("W", shape=[dim, self.hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(
                seed=config.RANDOM_SEED))
            b = tf.get_variable("b", shape=[self.hidden_size])
            h = tf.nn.xw_plus_b(x, W, b)
            h_drop = tf.nn.dropout(tf.nn.relu(h), self.dropout_keep_prob,
                seed=config.RANDOM_SEED)
        return h_drop

    def learn_feature_embedding(self):
        batch_size = tf.shape(self.heads)[0]
        fe1 = tf.nn.embedding_lookup(self.features, self.heads)
        fe2 = tf.nn.embedding_lookup(self.features, self.tails)
        feature_embedding = tf.concat([fe1, fe2], axis=0)
        h_drop = tf.nn.dropout(feature_embedding, self.dropout_keep_prob,
            seed=config.RANDOM_SEED)
        for i in range(self.hidden_layers):
            h_drop = self.add_hidden_layer(h_drop, i)
        with tf.variable_scope("feature_embedding"):
            W = tf.get_variable("W", shape=[self.hidden_size, self.fe_size],
                initializer=tf.contrib.layers.xavier_initializer(
                seed=config.RANDOM_SEED))
            b = tf.get_variable("b", shape=[self.fe_size])
            h = tf.nn.xw_plus_b(h_drop, W, b)
        feat_e1, feat_e2 = tf.split(h, [batch_size, batch_size])
        return feat_e1, feat_e2

class FeatE_DistMult(FeatE):
    def __init__(self, n_entities, n_relations, hparams, features):
        super(FeatE_DistMult, self).__init__(n_entities, n_relations, hparams)
        self.l2_reg_lambda = hparams.l2_reg_lambda
        self.fe_size = hparams.fe_size
        self.feat_size = features.shape[1]
        self.hidden_layers = hparams.hidden_layers
        self.hidden_size = hparams.hidden_size
        self.dropout_keep_prob = hparams.dropout_keep_prob
        self.features = tf.Variable(features, trainable=False, dtype=tf.float32)
        self.build()

    def add_params(self):
        self.entity_embedding = tf.Variable(tf.random_uniform([self.n_entities,
            self.embedding_size], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="entity_embedding")
        self.relation_embedding = tf.Variable(tf.random_uniform([self.n_relations,
            (self.embedding_size + self.fe_size)], 0., 1., seed=config.RANDOM_SEED),
            dtype=tf.float32, name="relation_embedding")

    def add_prediction_op(self):
        feat_e1, feat_e2 = self.learn_feature_embedding()
        free_e1 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding, self.heads))
        free_e2 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding, self.tails))
        self.e1 = tf.concat([free_e1, feat_e1], axis=1)
        self.e2 = tf.concat([free_e2, feat_e2], axis=1)
        self.r = tf.nn.embedding_lookup(self.relation_embedding, self.relations)

        self.pred = tf.nn.sigmoid(tf.reduce_sum(self.e1 * self.r * self.e2, -1),
                name="pred")

    def add_loss_op(self):
        losses = tf.nn.softplus(-self.labels * tf.reduce_sum(
            self.e1 * self.r * self.e2, -1))
        self.l2_loss = tf.reduce_mean(tf.square(self.e1)) + \
            tf.reduce_mean(tf.square(self.e2)) + \
            tf.reduce_mean(tf.square(self.r))
        self.loss = tf.add(tf.reduce_mean(losses), self.l2_reg_lambda *
            self.l2_loss, name="loss")
