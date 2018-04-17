import tensorflow as tf
import numpy as np
import pickle
from layers import embeddings_layer
from layers import mlp_layer


class WordPairClassifier(object):
	"""
	A model for classifying relations between pairs of words by means of (multiple) non-linear (MLP) transformation (tensor-MLP) of input word representations
	"""
	### Initializing the word-pair prediction model
	### Parameters: 
	###		- embeddings - a numpy matrix containing the embeddings of all vocabulary words
	###		- embedding_size - size of the input word embeddings
	###		- same_mlp - indicates whether to use the same (or different if set to False) non-linear transformation for both words in the pair 
	###	    - 
	def __init__(self, embeddings, embedding_size, mlp_hidden_layer_sizes, same_mlp = True, bilinear_softmax = False, num_mappings = 1, activation = tf.nn.tanh, num_classes = 2, noise_std = 0.01, scope = "word_pair_model", dist_labels = None):
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.scope = scope
		self.same_mlp = same_mlp
		self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
		self.num_mlps = num_mappings
		self.bilinear_softmax = bilinear_softmax
		self.num_classes = num_classes
		self.noise_std = noise_std	
		self.dist_labels = dist_labels	

		# placeholders (first dimension, marked with None is the batch size)
		# input_w1 - indices of first words in pairs, in the embeddings vocabulary
		# input_w2 - indices of second words in pairs, in the embeddings vocabulary
		# dropout - placeholder for putting the value of dropout rate
		with tf.name_scope(self.scope + "__placeholders"):
			self.input_w1 = tf.placeholder(tf.int32, [None,], name="w1")
			self.input_w2 = tf.placeholder(tf.int32, [None,], name="w2")
			self.dropout = tf.placeholder(tf.float64, name="dropout")
			self.training = tf.placeholder(tf.bool, name="training")
		
		# defining the word embedding layer (the input embeddings are not updateable)
		print("Defining embeddings layer...")
		self.emb_layer = embeddings_layer.EmbeddingLayer(None, self.embeddings, embedding_size, update_embeddings = False)
		
		# looking up word embeddings from their vocabulary indices
		print("Embedding lookup and noisification...")
		self.embs_w1 = self.emb_layer.lookup(self.input_w1) #tf.cond(self.training, lambda: noise.add_gaussian_noise_layer(self.emb_layer.lookup(self.input_w1), self.noise_std), lambda: self.emb_layer.lookup(self.input_w1))
		self.embs_w2 = self.emb_layer.lookup(self.input_w2) #tf.cond(self.training, lambda: noise.add_gaussian_noise_layer(self.emb_layer.lookup(self.input_w2), self.noise_std), lambda: self.emb_layer.lookup(self.input_w2))

		# MLPs (with or without shared parameters, depending on same_mapper)
		print("Defining FFNs (MLPs)...")
		self.left_mappers = []
		for i in range(self.num_mlps):
			print("mlp number #" + str(i+1))
			mlp_left = mlp_layer.MultiLayerPerceptron(mlp_hidden_layer_sizes, embedding_size, scope = "mlp_" + str(i) + ("" if same_mlp else "_first"), unique_scope_addition = "_1")
			mlp_left.define_model(activation = activation, previous_layer = self.embs_w1, share_params = None) 
			self.left_mappers.append(mlp_left)
		
		self.right_mappers = []
		for i in range(self.num_mlps):
			mlp_right = mlp_layer.MultiLayerPerceptron(mlp_hidden_layer_sizes, embedding_size, scope = "mlp_" + str(i) + ("" if same_mlp else "_second"), unique_scope_addition = "_2")
			mlp_right.define_model(activation = activation, previous_layer = self.embs_w2, share_params = same_mlp)
			self.right_mappers.append(mlp_right)

		# VARIANT 1 of combining the specialized embeddings 
		if self.bilinear_softmax:
			print("Defining bilinear products...")
			#if self.num_mlps != self.num_classes:
			#	raise ValueError("Model is incorrectly configured -- bilinear softmax variant of the model must have the same number of MLPs as the number of prediction classes!")

			self.bilinear_Ws = []
			self.bilinear_bs = []
			all_bilin_prods = []
			for i in range(self.num_mlps):
				print("Bilinear product #" + str(i+1))
				with tf.variable_scope(self.scope + "__variables"):
					bilinear_W = tf.get_variable("W_bilin_" + str(i), shape=[self.left_mappers[i].output_size, self.right_mappers[i].output_size], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
					bilinear_b = tf.get_variable("b_bilin_" + str(i), initializer = tf.constant(0.1, shape=[1], dtype = tf.float64), dtype = tf.float64)
					self.bilinear_Ws.append(bilinear_W)
					self.bilinear_bs.append(bilinear_b)
	
				lin_part = tf.matmul(self.left_mappers[i].outputs, bilinear_W)
				bilin_part = tf.add(tf.reduce_sum(tf.multiply(lin_part, self.right_mappers[i].outputs), axis = -1), bilinear_b)

				### for BILIN-PROD model ###
				#lin_part = tf.matmul(self.embs_w1, bilinear_W)
				#bilin_part = tf.add(tf.reduce_sum(tf.multiply(lin_part, self.embs_w2), axis = -1), bilinear_b)

				bilin_score = activation(bilin_part)
				all_bilin_prods.append(tf.nn.dropout(bilin_score, self.dropout))
			
			rel_outs = tf.transpose(tf.stack(all_bilin_prods))
			#rel_outs = batch_norm.batch_normalization(tf.transpose(tf.stack(all_bilin_prods)), self.scope + "_bn_bilin_scores", self.training, False)           		

			with tf.variable_scope(self.scope + "__variables"):
				self.W_class = tf.get_variable("W_class", shape=[self.num_mlps, self.num_classes], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
				self.b_class = tf.get_variable("b_class", initializer = tf.constant(0.1, shape=[self.num_classes], dtype = tf.float64), dtype = tf.float64)
			out_scores = activation(tf.matmul(rel_outs, self.W_class) + self.b_class)
			self.outputs = out_scores
			#self.outputs = batch_norm.batch_normalization(out_scores, self.scope + "_bn_out", self.training, False)
		
		# VARIANT 2 of combining the specialized embeddinfs 
		else:
			print("Defining concats and linear classifier...")
			outs_left = tf.concat([x.outputs for x in self.left_mappers], 1)
			outs_right = tf.concat([x.outputs for x in self.right_mappers], 1)
			pair_concat_out = tf.concat([outs_left, outs_right], 1)
			
			with tf.variable_scope(self.scope + "__variables"):
				self.W_class = tf.get_variable("W_class", shape=[self.mlp_hidden_layer_sizes[-1] * self.num_mlps * 2, self.num_classes], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
				self.b_class = tf.get_variable("b_class", initializer = tf.constant(0.1, shape=[self.num_classes], dtype = tf.float64), dtype = tf.float64)
				self.outputs = activation(tf.matmul(pair_concat_out, self.W_class) + self.b_class)

		# predictions
		self.preds = self.outputs
		self.preds_raw = tf.nn.softmax(self.outputs)

		# L2-regularization loss
		self.l2_loss = 0 
		print("Defining L2 loss...")
		for i in range(self.num_mlps):
			self.l2_loss += self.left_mappers[i].l2_loss if same_mlp else (self.left_mappers[i].l2_loss + self.right_mappers[i].l2_loss + self.right)
			if self.bilinear_softmax:
				self.l2_loss += tf.nn.l2_loss(self.bilinear_Ws[i]) + tf.nn.l2_loss(self.bilinear_bs[i])
		self.l2_loss += tf.nn.l2_loss(self.W_class) + tf.nn.l2_loss(self.b_class)

	### This method defines the loss function for the training
	### Parameters:
	###		- loss_function - the delegate to the actual loss computation function
	###		- l2_reg_factor - the factor for the regularization component (sum of norms of parameter matrices and vectors)
	###		- learning_rate - initial learning rate
	###		- loss_function_params - additional parameters for the computation of the loss function
	def define_optimization(self, loss_function, l2_reg_factor = 0.01, learning_rate = 1e-3, loss_function_params = None):
		print("Defining loss...")
		with tf.name_scope(self.scope + "__placeholders"):
			self.input_y = tf.placeholder(tf.float64, [None, self.num_classes], name="input_y")
		if loss_function_params:
			self.pure_loss = loss_function(self.outputs, self.input_y, loss_function_params)
		else:
			self.pure_loss = loss_function(self.outputs, self.input_y)
		self.loss = self.pure_loss + l2_reg_factor * self.l2_loss
			
		print("Defining optimizer...")
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		print("Done!...")
				
	def get_feed_dict(self, left_words, right_words, labels, dropout, training):
		if labels: 
			fd = { self.input_w1 : left_words, self.input_w2 : right_words, self.input_y : labels, self.dropout : dropout, self.training : training}
		else:
			fd = { self.input_w1 : left_words, self.input_w2 : right_words, self.dropout : dropout, self.training : training}
		for i in range(self.num_mlps):
			fd.update(self.left_mappers[i].get_feed_dict(None, None, dropout))
			fd.update(self.right_mappers[i].get_feed_dict(None, None, dropout))
		return fd
	def get_variable_values(self, session):
		variables = []
		for i in range(self.num_mlps):
			vars_left_i = self.left_mappers[i].get_variable_values(session)
			if not self.same_mlp:
				vars_right_i = self.right_mappers[i].get_variable_values(session)
			W_i = self.bilinear_Ws[i].eval(session = session)
			b_i = self.bilinear_bs[i].eval(session = session)
			if self.same_mlp: 
				variables.append([vars_left_i, W_i, b_i]) 
			else:
				variables.append([vars_left_i, vars_right_i, W_i, b_i]) 
		W_cl = self.W_class.eval(session = session)
		b_cl = self.b_class.eval(session = session)
		variables.append(W_cl)
		variables.append(b_cl)
		return variables

	def set_variable_values(self, session, variables):
		for i in range(self.num_mlps):
			if self.same_mlp:
				vars_left_i, W_i, b_i = variables[i]
			else:
				vars_left_i, vars_right_i, W_i, b_i = variables[i]
			
			self.left_mappers[i].set_variable_values(session, vars_left_i)
			self.right_mappers[i].set_variable_values(session, vars_left_i if self.same_mlp else vars_right_i)
			session.run(self.bilinear_Ws[i].assign(W_i))
			session.run(self.bilinear_bs[i].assign(b_i))
		session.run(self.W_class.assign(variables[-2]))
		session.run(self.b_class.assign(variables[-1]))

	def get_hyperparameters(self):
		return [self.same_mlp, self.bilinear_softmax, self.mlp_hidden_layer_sizes, self.num_mlps, self.num_classes, self.embedding_size, self.dist_labels]

	def get_model(self, session):
		hyperparams = self.get_hyperparameters()
		variables = self.get_variable_values(session)
		return (hyperparams, variables)
