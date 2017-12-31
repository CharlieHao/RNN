#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: NLP program:
#               Use word embedding to map one hot encoding word vector of size V to a vector of size D
# 				Use word embedding as input of RNN
# 				Train the embedding process by setting embedding layer
# Method: RNN:  case 3: each sequence: input:[START,w1,w2,w3...],target:[w1,w2,w3,....]
# 				a sequence of input, x_t+1 is the output of x_t		 			  
# Data structure: D: dimensionality of word embedding 
#				  V: vocabulary size (size of the word2idx dictionary)
#				  M: hidden layer size 
# Tips: This is not a general structure for RNN
#		There is a word embedding layer
#		General case just need to build a structure without embedding layer, so first layer is the input layer 	
#		This structure is the basic structure: Withput MLP structure in hidden layer
#											   recurrence happened directly through hidden layer			  


import tensorflow as tf 
import numpy as np 
import matplotlib as plt 

from sklearn.utils import shuffle
from utils import init_params, get_robert_frost, get_wikipedia_data

class RNN_with_embedding(object):
	def __init__(self,D,M,V,f,session):
		# f: activation function in the d=hidden layer
		self.D = D
		self.M = M
		self.V = V
		self.f = f
		self.session = session

	def init_session(self,session):
		self.session = session

	# this coding structure make it convenient to edit, but can not adjust params in optimization
	def structure(self,W_e,W_x,W_h,b_h,h0,W_o,b_o):
		'''
		W_e: embedding matrix, V * D
		other parameters are in the RNN structure
		'''
		# build tensorflow structure
		self.W_e = tf.Variable(W_e)
		self.W_x = tf.Variable(W_x)
		self.W_h = tf.Variable(W_h)
		self.W_o = tf.Variable(W_o)
		self.b_h = tf.Variable(b_h)
		self.b_o= tf.Variable(b_o)
		self.h0 = tf.Variable(h0)
		self.params = [self.W_e,self.W_x,self.W_h,self.b_h,self.h0,self.W_o,self.b_o]

		self.tfX = tf.placeholder(tf.int32,shape=(None,),name='X') # it seems that int32 is enough
		self.tfY = tf.placeholder(tf.int64,shape=(None,),name='Y')

		# do one hot coding for a input tensor: a sequence of index of word in the word2idx dict
		# X_W is the tensor matrix (D space) corresponding to an input tensor(sequence of index of word)
		X_W = tf.nn.embedding_lookup(W_e,self.tfX)

		def recurrence(hidden_tminus1,XW_t):
			# input X_t has a I matrix, means a direct effect on the hidden_t
			hidden_tminus1 = tf.reshape(hidden_tminus1,(1,self.M))
			hidden_t = self.f(XW_t+tf.matmul(hidden_tminus1,self.W_h))
			hidden_t = tf.reshape(hidden_t,(self.M,))
			return hidden_t

		hidden = tf.scan(
			fn=recurrence,
			elems=X_W,
			initializer=self.h0 
		)

		# From hidden layer to output
		logits = tf.matmul(hidden,self.W_o) + self.b_o

		# predict_op
		prediction = tf.argmax(logits,1)
		logits_softmax = tf.nn.softmax(logits)
		self.output_prob = logits_softmax
		self.predict_op = prediction

		# cost_op
		# we need to reshape, 
		# since in tf.nn.sampled_softmax_loss: inputs: [batch_size, dim]
		#									   labels: [batch_size, num_true]
		nce_weights = tf.transpose(self.W_o,[1,0])
		nce_bias = self.b_o
		hidden = tf.reshape(hidden,(-1,self.M))
		labels = tf.reshape(self.tfY,(-1,1))

		self.cost_op = tf.reduce_mean(
			tf.nn.sampled_softmax_loss(
				weights=nce_weights,
				biases=nce_bias,
				labels=labels,
				inputs=hidden,
				num_sampled=50,
				num_classes=self.V
			)
		)

		# train_op: try adam boost and momentum
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost_op)
		# self.train_op = tf.train.MomentumOptimizer(10e-4,.99).minimize(self.cost_op)

		# init_op
		self.init_op = tf.global_variables_initializer()

	def fit(self,X,epoches=1000,print_period=1,show_fig=False):
		# X should be a jagged array: N lines, each line is a sequence of index of word
		N = len(X)

		# initial weights 
		W_e = init_params(self.V,self.D).astype(np.float32)
		W_x = init_params(self.D,self.M).astype(np.float32)
		W_h = init_params(self.M,self.M).astype(np.float32)
		b_h = np.zeros(self.M).astype(np.float32)
		h0 = np.zeros(self.M).astype(np.float32)
		W_o = init_params(self.M,self.V).astype(np.float32)
		b_o = np.zeros(self.V).astype(np.float32)

		self.structure(W_e,W_x,W_h,b_h,h0,W_o,b_o)
		self.session.run(self.init_op)

		neg_loglikelihood = []
		total_len = sum(len(sequence)+1 for sequence in X)
		for ite in range(epoches):
			X = shuffle(X)
			correct = 0
			cost = 0
			for n in range(N):
				input_seq = [0] + X[n]
				target_seq = X[n] + [1]
				
				self.session.run(
					self.train_op,
					feed_dict={self.tfX:input_seq,self.tfY:target_seq}
				)
				c = self.session.run(
					self.cost_op,
					feed_dict={self.tfX:input_seq,self.tfY:target_seq}
				)
				prediction = self.session.run(
					self.predict_op,
					feed_dict={self.tfX:input_seq}
				)
				cost+=c

				for pred, targ in zip(prediction,target_seq):
					if pred == targ:
						correct += 1

			neg_loglikelihood.append(cost)
			print('iteration:',ite,'correct rate:',correct/total_len)

	def predict_distribution(self,pre_words):
		return self.session.run(
			self.output_prob,
			feed_dict = {self.tfX:pre_words}
		)

	def save_files(self,filename):
		saved_parameters = self.session.run(self.params)
		np.savez(filename, *[p for p in saved_parameters])

	@ staticmethod
	def load(filename,activation_function,session):

		npz = np.load(filename)
		W_e = npz['arr_0']
		W_x = npz['arr_1']
		W_h = npz['arr_2']
		b_h = npz['arr_3']
		h0 = npz['arr_4']
		W_o = npz['arr_5']
		b_o = npz['arr_6']

		V,D = W_e.shape
		_,M = W_x.shape
		rnn = RNN_with_embedding(D,M,V,activation_function,session)
		rnn.structure(W_e,W_x,W_h,b_h,h0,W_o,b_o)
		return rnn

	def generate(self, pi, word2idx):
		idx2word = {v:k for k,v in word2idx.iteritems()}
		V = len(pi)

		n_lines = 0


		X = [ np.random.choice(V, p=pi) ]
		print(idx2word[X[0]],)

		while n_lines < 4:
			probs = self.predict(X)[-1]
			word_idx = np.random.choice(V, p=probs)
			X.append(word_idx)
			if word_idx > 1:
				word = idx2word[word_idx]
				print (word,)
			elif word_idx == 1:
				n_lines += 1
				print ('')
				if n_lines < 4:
					X = [ np.random.choice(V, p=pi) ] # reset to start of line
					print(idx2word[X[0]],)





























