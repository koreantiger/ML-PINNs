"""
Conducts PIML as described in Fuck & Tchelepi 2020
WARNING theta in the conservation eqquation is not equal to 1 here.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from pyDOE import lhs

np.random.seed(1234)
# uncomitt

# tf.set_random_seed(1234)

# %%
def plotbenchmark(t1, t_p, Exact, U_pred):
    # compare to analytical solultion
    idx = np.abs(t_p - t1).argmin()
    plt.figure(2)
    plt.grid()
    plt.title('t={:.2f}'.format(t1))
    print(t1)
    print(t_p[idx])
    plt.plot(x, Exact[idx , :], '-k', linewidth = 4,label='exact')
    plt.plot(x, U_pred[-1, :], '--r', linewidth = 4,label='prediction')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

# def plotbenchmark(t1, Exact, U_pred):
#     # compare to implicit solver
#     z = np.linspace(0,1,100)
#     xx = np.loadtxt('a.csv', dtype='float16', delimiter=",")
#     t_p = np.linspace(0,1,100)
#     idx = np.abs(t_p - t1).argmin()
#     plt.figure(2)
#     plt.grid()
#     plt.title('t={:.2f}'.format(t1))
#     print(t1)
#     print(t_p[idx])
#     plt.plot(z, xx[idx+7 , :], '-k', label='exact')
#     plt.plot(x, U_pred[-1, :], '--r', label='prediction')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.legend()
#     plt.show()
def plotstdtraining(t1, t_p, Exact, U_pred):
    t = np.arange(0, 1, 0.1)
    t_p = np.linspace(0,1,data.shape[1])
    t0 = 0
    # sequential traininf
    plt.figure()
    plt.grid()
    for i in range(len(t)):
        t1 = t0 + dt
        idx = np.abs(t_p - t1).argmin()
        plt.figure(2)
        plt.title('t={:.2f}'.format(t1))
        print(t1)
        print(t_p[idx])
        plt.plot(x, Exact[idx , :], '-k', label='exact')
        plt.plot(x, U_pred[idx, :], '--r', label='prediction')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
        t0 += dt

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub
        self.X_u = X_u
        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]
        self.X_f = X_f
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        """
                    tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                       self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        """
        self.loss = tf.reduce_mean(tf.square(tf.math.abs(self.u_tf - self.u_pred))) + \
                    tf.reduce_mean(tf.square(tf.math.abs(self.f_pred)))

        # bc + ic + collocation point
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
        #                                                         method='L-BFGS-B',
        #                                                         options={'maxiter': 20000})
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 60000})
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,amsgrad=False, name='Adam')


        # 'maxfun': 50000,
        # 'maxcor': 50,
        # 'maxls': 50,
        # 'ftol' : 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.loss_val = []

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        f_x = tf.gradients(u ** 2 / (u ** 2 + 1*(1 - u) ** 2), x)[0]
        f = u_t + 0.5555555555555556 * f_x
        #f = u_t + f_x
        return f

    def callback(self, loss):
        self.loss_val.append(loss)
        print('Loss: %e' % (loss))

    def train(self, nIter):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}


        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star



    def evaluate_boundary_conditions(self):
        nt = np.linspace(0, 1, 16)
        dt = max(np.diff(nt))
        t = 0
        plt.figure(figsize=(10, 5))
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.colorbar(h)
        plt.xlabel('t')
        plt.ylabel('x')

        for x in range(len(nt)):
            idx = np.where((t < self.X_u[:, 1]) & (self.X_u[:, 1] < t + dt))
            plt.scatter(self.X_u[idx, 1], self.X_u[idx, 0], color='r', marker='.', label='u')
            t += dt
        plt.show()
        """
        for x in range(len(nt)):
            t += dt
             u_pred, f_pred = model.predict(X_temp)

        """

    def plot_data_training_seq(self):
        plt.figure(figsize=(10, 5))
        plt.xlim([-0.1, 1.1]);
        plt.ylim([-0.1, 1.1])
        plt.colorbar(h)
        plt.xlabel('t')
        plt.ylabel('x')
        t = 0
        nt = np.linspace(0, 1, 16)
        dt = max(np.diff(nt))
        for x in range(len(nt)):
            # sequentially training
            idx = np.where((t < self.X_f[:, 1]) & (self.X_f[:, 1] < t + dt))
            self.x_f = self.X_f[idx, 1:2]
            self.t_f = self.X_f[idx, 0:1]
            plt.scatter(self.X_f[idx, 1:2], self.X_f[idx, 0:1], color='r', marker='.', label='u')
            plt.pause(0.1)
            t += dt

        plt.show()


# %%

if __name__ == "__main__":
    N_u = 500  # boundary and initial conditions points
    # N_f = 10000  # collocation points for the residual term of the loss function
    N_f = 5000  # collocation points for the residual term of the loss function
    #N_f = 5000  # collocation points for the residual term of the loss function
    #N_f = 10000  # collocation points for the residual term of the loss function

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]  # neurons per layer
    #layers = [2, 20, 20, 20, 20, 20, 1]  # neurons per layer
    #layers = [2, 20, 20, 20, 20,20,20,20, 20, 1]  # neurons per layer
    data = np.load('PIML_data.npy')

    # t = np.linspace(0, 1, data.shape[1])  # time from 0 to 1
    # x = np.linspace(0, 1, data.shape[0])  # position from 0 to 1

    dt = 0.1  # interval of time
    t = np.arange(0, 1, dt)
    t_p = np.linspace(0,1,data.shape[1])
    t0 = 0
    update = False
    # sequential training
    for i in range(len(t)):
        t1 = t0 + dt
        Exact = data.T  # exact solution u(x,t) determined from fully impliict solver
        t = np.linspace(t0, t1, data.shape[1])
        x = np.linspace(0, 1, data.shape[0])
        X, T = np.meshgrid(x, t)
        X_temp = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_temp = Exact.flatten()[:, None]

        # Domain bounds for normalisation
        lb = X_temp.min(0)
        ub = X_temp.max(0)

        # identify initial and boundary conditions points for training set

        # if (i == 0):
        #     # Ɐx and t=0 -> u(x,t) = 0
        #     xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        #     uu1 = Exact[0:1, :].T
        # else:
        #     # Ɐx and t=t0 + dt -> u(x,t) = 0
        #     xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        #     # uu1 = Exact[0:1, :].T
        #     lxxx = len(U_pred[-1, :])
        #     uu1 = U_pred[-1, :].reshape(lxxx, 1)
        if (i == 0):
            # Ɐx and t=0 -> u(x,t) = 0
            xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
            uu1 = Exact[0:1, :].T
        elif(update==True):
            # Ɐx and t=t0 + dt -> u(x,t) = 0
            xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
            # uu1 = Exact[0:1, :].T
            lxxx = len(U_pred[-1, :])
            uu1 = U_pred[-1, :].reshape(lxxx, 1)

        # uu1 = np.zeros(np.shape(Exact[0:1,:].T))
        # x = 0 and t>0 -> u(x,t) = 1
        xx2 = np.hstack((X[1:, 0:1], T[1:, 0:1]))
        uu2 = Exact[1:, 0:1]

        X_u_train = np.vstack([xx1, xx2])
        u_train = np.vstack([uu1, uu2])

        # randomly sample u-set for training data set
        idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx, :]

        # find collocation points
        X_f_train = lb + (ub - lb) * lhs(2, N_f)


        # construct NN and train
        model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
        model.train(0)
        # model.evaluate_boundary_conditions()
        # model.plot_data_training_seq()
        u_pred, f_pred = model.predict(X_temp)
        error_u = np.linalg.norm(u_temp - u_pred, 2) / np.linalg.norm(u_temp, 2)
        U_pred = griddata(X_temp, u_pred.flatten(), (X, T), method='linear')
        idx = np.abs(t_p - t1).argmin()
        plotbenchmark(t1,t_p, Exact, U_pred)
        # dynamic timestep training
        if model.loss_val[-1] < 3.5e-3:
            t0 += dt
            dt*=2
            N_f= int(N_f)
            update = True
        else:
            dt/=2
            N_f/=2
            N_f= int(N_f)
            if model.loss_val[-1] == model.loss_val[-1]:
                print('local minimum')
            update = False
                #model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)


def plotbenchmark(t1, t_p, Exact, U_pred):
    idx = np.abs(t_p - t1).argmin()
    plt.figure(2)
    plt.title('t={:.2f}'.format(t1))
    plt.plot(x, Exact[idx, :], '-k', label='exact')
    plt.plot(x, U_pred[-1, :], '--r', label='prediction')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
# %%
sample = np.arange(0, 736, 100)
#
for i in sample:
    plt.figure()
    plt.grid()
    plt.title('t={:.2f}'.format(t[i]))
    plt.plot(x, Exact[i, :], '-k', label='exact')
    plt.plot(x, U_pred[i, :], '--r', label='prediction')
    plt.xlim([0, 1]);
    plt.ylim([0, 1])
    plt.legend()
    plt.show()