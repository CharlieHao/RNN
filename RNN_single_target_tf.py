#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: Parity project:
#               based on XOR logic gate
# 				Used for detection of the error message in communication system
# Method: RNN: 	case 1
# 			    a sequence of input and a single vector output at time T		 			  



import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from utils import init_params, parity_data, parity_data_labeled

import tensorflow as tf 
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
from tensorflow.contrib.rnn import static_rnn 

class RNN(object):
	def __init__(self,M):
		self.M = M

	def reshapedim(self,x,T,D,batch_size):
		x = tf.transpose(x,(1,0,2))
		x = tf.reshape(x,(batch_size*T,D))
		x = tf.split(x,T)
		# After that, we can get num = T of tensors, each tensor is of size of batch_size
		return x

	def fit(self,X,Y,activation_func=tf.nn.relu, batch_size=10, Cell=BasicRNNCell,
		learning_rate=10e-1,mu=.99,epochs=1000,print_period=10,show_fig=True):
		# batch_size should be equal to the batch_size of reshapedim function
		# Cell type could also be GRUCell
		# activation_func also could be softmax, sigmoid .....
		N,T,D = X.shape
		M = self.M
		K = len(set(Y.flatten()))
		# initialize the parameters 	
		W = init_params(M,K).astype(np.float32)
		b = np.zeros(K,dtype = np.float32)
		# construct the tensorflow structure
		self.W = tf.Variable(W)
		self.b = tf.Variable(b)

		tfX = tf.placeholder(tf.float32,shape=(batch_size,T,D),name='inputs')
		tfY = tf.placeholder(tf.int64,shape=(batch_size,T),name='targets')

		# generate tensors 
		X_sequence= self.reshapedim(tfX,T,D,batch_size)

		# build the RNN Cell: Basic or GRU
		rnn_cell = Cell(num_units=self.M,activation=activation_func)
		# outputs, states = rnn_module.rnn(rnn_unit, sequenceX, dtype=tf.float32)
		# this is the ∂=h(a) (a_t =Ws_t-1+W'x_t+b) and hidden state s of the RNN cell
		# IMPORTANT: this result corresponding to case 2, and we get all of the hidden states sequence and target prediction sequence
		outputs,states = static_rnn(rnn_cell,X_sequence,dtype=tf.float32)
		self.outputs = outputs
		self.states = states
		# outputs shape is (batch_size,T,M), need to be reshaped
		outputs = tf.transpose(outputs, (1,0,2))
		outputs = tf.reshape(outputs,(batch_size*T,M))
		labels = tf.reshape(tfY,(batch_size*T,))
		# build the cost_op predict_op and train_op
		logits = tf.matmul(outputs,self.W)+self.b
		
		predict_op = tf.argmax(logits,1)
		cost_op = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels = labels,
				logits = logits
			)
		)
		train_op = tf.train.MomentumOptimizer(learning_rate,momentum=mu).minimize(cost_op)

		# combine the tensorflow structure and data
		costs = []
		n_batches = int(N/batch_size)

		init_op = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init_op)
			for n in range(epochs):
				X,Y = shuffle(X,Y)
				correct = 0
				cost = 0
				for i in range(n_batches):
					Xbatch = X[i*batch_size:i*batch_size+batch_size]
					Ybatch = Y[i*batch_size:i*batch_size+batch_size]

					session.run(train_op,feed_dict={tfX: Xbatch, tfY: Ybatch})
					c = session.run(cost_op,feed_dict={tfX: Xbatch, tfY: Ybatch})
					p = session.run(predict_op,feed_dict={tfX: Xbatch, tfY: Ybatch})

					cost += c
					# This step correspont to case 1
					for j in range(batch_size):
						index = (j+1)*T-1 # the last of each sequence, since only one observation y
						correct += (p[index]==Ybatch[j][-1])

				if n%print_period==0:
					print('iteration:',n,'cost:',cost,'classification rate:',correct/N)

				if correct==N:
					print('iteration:',n,'cost:',cost,'classification rate:',correct/N)
					break
				costs.append(cost)

			if show_fig:
				plt.plot(costs)
				plt.show()


















