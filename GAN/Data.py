import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt


class DataDistribution:
    def __init__(self, mu=3, sigma=0.5):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class Generator:
    def __init__(self, ran):
        self.ran = ran

    def sample(self, N):
        samples = np.linspace(-self.ran, self.ran, N) + \
            np.random.random(N) * 0.01
        return samples


def linear(input_data, output_dim, scope=None, stddev=1):
    nor = tf.random_normal_initializer(stddev=stddev)
    cons = tf.constant_initializer(.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input_data.shape[1], output_dim],
                            initializer=nor)
        b = tf.get_variable('b', [output_dim], initializer=cons)
    return tf.matmul(input_data, w) + b


def generator(input_data, output_dim):
    h0 = tf.nn.softplus(linear(input_data, output_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input_data, output_dim):
    h0 = tf.nn.tanh(linear(input_data, output_dim*2, 'd0'))
    h1 = tf.nn.tanh(linear(h0, output_dim*2, 'd1'))
    h2 = tf.nn.tanh(linear(h1, output_dim*2, 'd2'))
    h3 = tf.nn.sigmoid(linear(h2, 1, 'd3'))
    return h3


def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    batch = tf.Variable(0)
    num_decay_steps = 150
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               batch,
                                               num_decay_steps,
                                               decay,
                                               staircase=True)
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimize


class GAN:
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.learning_rate = 0.03
        self.mlp_hidden_size = 4
        self._create_model()

    def _create_model(self):
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, 1))
            self.pre_label = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_label))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        with tf.variable_scope('Generator'):
            self.z = tf.placeholder(tf.float32,
                                    shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        with tf.variable_scope('Discriminator') as scope:
            self.x = tf.placeholder(tf.float32,
                                    shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.mlp_hidden_size)

        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(tf.log(1 - self.D2))

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='Discriminator')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='Generator')
        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)
        print('model created successfully')

    def train(self):
        print('start training')
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            num_pre_training_steps = 1000
            for step in range(num_pre_training_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pre_train_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1),),
                    self.pre_label: np.reshape(labels, (self.batch_size, 1))
                })
            weightsD = session.run(self.d_pre_params)
            for i, v in enumerate(self.d_params):
                session.run(v.assign(weightsD[i]))
            plt.ion()
            f, ax = plt.subplots(1)
            ax.set_ylim(0, 1)
            plt.title('GAN Visualization')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            real = generated = None
            for step in range(self.num_steps):
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })
                if step % self.log_every == 0:
                    print('{}:{}\t{}'.format(step, loss_d, loss_g))
                    if step % 100 == 0 or step == 0 or step == self.num_steps - 1:
                        real, generated = self._plot_distribution(session, real, generated)
                        plt.legend()
            plt.ioff()

    def _sample(self, session, num_points=1000, num_bins=100):
        # xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.ran, self.gen.ran, num_bins)
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)
        zs = np.linspace(- self.gen.ran, self.gen.ran, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size*i: self.batch_size*(i+1)] = session.run(self.G, {
                self.z: np.reshape(zs[self.batch_size*i: self.batch_size*(i+1)],
                                   (self.batch_size, 1))
            })
        pg, _ = np.histogram(g, bins=bins, density=True)
        return pd, pg

    def _plot_distribution(self, session, real=None, generated=None):
        pd, pg = self._sample(session)
        p_x = np.linspace(-self.gen.ran, self.gen.ran, len(pd))
        if real is not None:
            r = real.pop(0)
            r.remove()
        real = plt.plot(p_x, pd, label='Real Data', color='red')
        if generated is not None:
            g = generated.pop(0)
            g.remove()
        generated = plt.plot(p_x, pg, label='Generated Data', color='blue')
        plt.pause(0.1)
        plt.show()
        return real, generated


if __name__ == '__main__':
    model = GAN(DataDistribution(),
                Generator(ran=8),
                2000,
                12,
                10)
    model.train()
