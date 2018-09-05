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
		hg1 = tf.layers.dense(X, hsize[0], activation=tf.nn.relu)
		hg2 = tf.layers.dense(hg1, hsize[1], activation=tf.nn.relu)
		hg3 = tf.layers.dense(hg2, 2)
		out = tf.layers.dense(hg3, 1)
	return out, hg3


if __name__ == '__main__':
	batch_size = 256
	ng = 10
	nd = 10

	X = tf.placeholder(tf.float32,[None, 2])
	Z = tf.placeholder(tf.float32,[None, 2])

	G = generator(Z)
	D_x_logits, D_x = discriminator(X)
	D_g_logits, D_g = discriminator(G, reuse=True)

	d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x_logits, labels=tf.ones_like(D_x_logits)) +
							tf.nn.sigmoid_cross_entropy_with_logits(logits=D_g_logits, labels=tf.zeros_like(D_g_logits)))
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_g_logits, labels=tf.ones_like(D_g_logits)))

	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
	d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")

	g_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(g_loss, var_list=g_vars) # G Train step
	d_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(d_loss, var_list=d_vars) # D Train step

	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	x_plot = sample_data(n=batch_size)

	for i in range(10001):
	    X_batch = sample_data(n=batch_size)
	    Z_batch = sample_Z(batch_size, 2)

	    for _ in range(nd):
	    	_, dloss = sess.run([d_step, d_loss], feed_dict={X: X_batch, Z: Z_batch})
	    rrep_dstep, grep_dstep = sess.run([D_x, D_g], feed_dict={X: X_batch, Z: Z_batch})
	    for _ in range(ng):
	    	_, gloss = sess.run([g_step, g_loss], feed_dict={Z: Z_batch})
	    rrep_gstep, grep_gstep = sess.run([D_x, D_g], feed_dict={X: X_batch, Z: Z_batch})

	    if not i % 100:
	    	print "Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i, dloss, gloss)


	    if i%500 == 0:
	        plt.figure()
	        g_plot = sess.run(G, feed_dict={Z: Z_batch})
	        xax = plt.scatter(x_plot[:,0], x_plot[:,1])
	        gax = plt.scatter(g_plot[:,0],g_plot[:,1])

	        plt.legend((xax,gax), ("Real Data","Generated Data"))
	        plt.title('Samples at Iteration %d'%i)
	        plt.tight_layout()
	        plt.savefig('./plots/iterations/iteration_%d.png'%i)
	        plt.close()

	        plt.figure()
	        rrd = plt.scatter(rrep_dstep[:,0], rrep_dstep[:,1], alpha=0.5)
	        rrg = plt.scatter(rrep_gstep[:,0], rrep_gstep[:,1], alpha=0.5)
	        grd = plt.scatter(grep_dstep[:,0], grep_dstep[:,1], alpha=0.5)
	        grg = plt.scatter(grep_gstep[:,0], grep_gstep[:,1], alpha=0.5)


	        plt.legend((rrd, rrg, grd, grg), ("Real Data Before G step","Real Data After G step",
	                               "Generated Data Before G step","Generated Data After G step"))
	        plt.title('Transformed Features at Iteration %d'%i)
	        plt.tight_layout()
	        plt.savefig('./plots/features/feature_transform_%d.png'%i)
	        plt.close()

	        plt.figure()

	        rrdc = plt.scatter(np.mean(rrep_dstep[:,0]), np.mean(rrep_dstep[:,1]),s=100, alpha=0.5)
	        rrgc = plt.scatter(np.mean(rrep_gstep[:,0]), np.mean(rrep_gstep[:,1]),s=100, alpha=0.5)
	        grdc = plt.scatter(np.mean(grep_dstep[:,0]), np.mean(grep_dstep[:,1]),s=100, alpha=0.5)
	        grgc = plt.scatter(np.mean(grep_gstep[:,0]), np.mean(grep_gstep[:,1]),s=100, alpha=0.5)

	        plt.legend((rrdc, rrgc, grdc, grgc), ("Real Data Before G step","Real Data After G step",
	                               "Generated Data Before G step","Generated Data After G step"))

	        plt.title('Centroid of Transformed Features at Iteration %d'%i)
	        plt.tight_layout()
	        plt.savefig('./plots/features/feature_transform_centroid_%d.png'%i)
	        plt.close()

