import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


###############################################################
###############################################################
#### Data part
####
def func(x):
	return 10 + x**2


def sample_data(n=10000, scale=100):
	data = []

	x = scale * (np.random.random_sample(n, ) - .5)

	for i in x:
		data.append([i, func(i)])

	return np.array(data)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


###############################################################
###############################################################
#### Model part
####
def generator(Z, hsize=[16, 16], reuse=False):
	with tf.variable_scope("Generator", reuse=reuse):
		h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.relu)
		h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.relu)
		out = tf.layers.dense(h2, 2)

	return out


def discriminator(X, hsize=[16, 16], reuse=False):
	with tf.variable_scope("Discriminator", reuse=reuse):
		h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.relu)
		h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.relu)
		h3 = tf.layers.dense(h2, 2)
		out = tf.layers.dense(h3, 1)
	return out, h3


if __name__ == '__main__':
	batch_size = 100
	# plt.scatter(X[:, 0], X[:, 1])
	# plt.show()

	X = tf.placeholder(tf.float32,[None,2])
	Z = tf.placeholder(tf.float32,[None,2])

	G = generator(Z)
	D_x_logits, D_x = discriminator(X)
	D_g_logits, D_g = discriminator(G, reuse=True)

	d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x_logits, labels=tf.ones_like(D_x_logits) + 
		tf.nn.sigmoid_cross_entropy_with_logits(logits=D_g_logits, labels=tf.zeros_like(D_x_logits))))
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_g_logits, labels=tf.ones_like(D_g_logits)))

	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
	d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")

	g_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(g_loss, var_list = g_vars) # G Train step
	d_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(d_loss, var_list = d_vars) # D Train step


	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	batch_size = 256
	nd_steps = 10
	ng_steps = 10

	for i in range(20001):
	    X_batch = sample_data(n=batch_size)
	    Z_batch = sample_Z(batch_size, 2)
	    _, dloss = sess.run([d_step, d_loss], feed_dict={X: X_batch, Z: Z_batch})
	    _, gloss = sess.run([g_step, g_loss], feed_dict={Z: Z_batch})

	    if not i % 1000:
	    	print "Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss)


	z = sample_Z(200, 2)
	g = sess.run(G, feed_dict={Z: z})
	print g.shape
	plt.scatter(g[:, 0], g[:, 1])
	plt.show()