import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 

from RNN_case1_tf import RNN
from utils import init_params, parity_data, parity_data_labeled
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell


def main(B=12,learning_rate=10e-4,epochs=1000):
	X,Y = parity_data_labeled(B)

	rnn = RNN(4)
	rnn.fit(X,Y,
		batch_size=10,
		learning_rate=learning_rate,
		epochs=epochs,
		activation_func=tf.nn.sigmoid,
		show_fig = False,
		Cell = BasicRNNCell
	)

if __name__ == '__main__':
	main()