from darts.engines import value_vector, index_vector
from darts.models.physics.dead_oil_python import DeadOil
from darts.models.darts_model import DartsModel
import random as rn
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, initializers, losses, layers
from keras.callbacks import EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt

class Model(DartsModel):
    def __init__(self, n_points=32):
        # call base class constructor
        super().__init__()
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        # create physics
        self.min_p = 300
        self.max_p = 800
        self.min_z = 1e-8
        self.max_z = 1 - self.min_z
        self.physics = DeadOil(timer=self.timer,
                               physics_filename=r'C:\Working_DARTS\darts-models\decouple_velocity\adgprs\physics_egg.in',
                               n_points=n_points, min_p=self.min_p, max_p=self.max_p, min_z=self.min_z)
        self.n_points = n_points
        self.p_vec = np.linspace(0, 800, self.n_points)
        self.z_vec = np.linspace(0, 1e-12, self.n_points)
        self.ops = value_vector([0, 1, 2, 3, 4, 5])
        self.ops_der = value_vector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.blk_idx = index_vector([0])
        self.p_vec = np.linspace(self.min_p, self.max_p, n_points)
        self.z_vec = np.linspace(self.min_z, self.max_z, n_points)
        self.ops = value_vector([0, 1, 2, 3, 4, 5])
        self.ops_der = value_vector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.blk_idx = index_vector([0])
        self.n_points = np.array(self.physics.acc_flux_itor.axis_points, dtype=int)
        self.Np = self.n_points[0]
        self.nz = self.n_points[1] - 2
        self.NC = self.physics.n_components
        self.timer.node["initialization"].stop()


    def compute_operator(self):
        # this function store alpha and beta operators for nc component and return them.ìsss
        x = np.append(self.n_points, self.physics.n_components)
        self.alpha = np.zeros(x)
        self.beta = np.zeros(x)
        self.data = np.zeros(np.append(np.prod(self.n_points), self.physics.n_components))
        self.val = np.zeros(np.prod(self.n_points))
        # store operators alpha and beta
        ctr = 0
        for p in range(len(self.p_vec)):
            for z in range(len(self.z_vec)):
                state = value_vector([self.p_vec[p], self.z_vec[z]])
                self.data[ctr,:] = self.p_vec[p], self.z_vec[z]
                self.physics.acc_flux_etor.evaluate(state, self.ops)
                self.val[ctr] = self.ops[self.NC]
                ctr +=1
                for x in range(self.NC):
                    self.alpha[p, z, x] = self.ops[x]
                    self.beta[p, z, x] = self.ops[x + self.NC]





    def plot_2D_parameter_space(self, itor=None):
        # this function store alpha and beta operators for nc component and return them. plus it plots operators
        from mpl_toolkits.mplot3d import Axes3D
        self.compute_operator()
        # plotting alpha and beta operators
        P_vec, Z_vec = np.meshgrid(self.z_vec, self.p_vec)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Z_vec, P_vec, self.beta[:, :, 1], color='b')
        ax.set_xlabel(r'$\omega_1$')
        ax.set_ylabel(r'$\omega_2$')
        ax.set_zlabel(r'$\beta$')
        plt.show()


    def train_NN(self):

        # Required in order to have reproducible results from a specific random seed
        os.environ['PYTHONHASHSEED'] = '0'

        # Force tf to use a single thread
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        random_seed = 27

        np.random.seed(random_seed)
        rn.seed(random_seed)
        tf.set_random_seed(random_seed)



        n_valid = 400
        n_total = 1600 + n_valid
        random_seed = 27
        np.random.seed(random_seed)
        rn.seed(random_seed)
        tf.set_random_seed(random_seed)

        n_train = 800
        n_eval = n_total - n_train - n_valid

        hidden = 12
        act = 'sigmoid'

        model = Sequential()

        model.add(Dense(hidden, input_shape=(2,), \
                        kernel_initializer=initializers.he_normal(), \
                        bias_initializer='zeros', activation=act))

        model.add(Dense(1, \
                        kernel_initializer=initializers.he_normal(), \
                        bias_initializer='zeros', activation=act))

        sgd = optimizers.SGD(lr=0.03, momentum=0.9, decay=1e-6, nesterov=True)
        nadam = optimizers.Nadam(lr=0.085, schedule_decay=0.005)

        es = EarlyStopping(monitor='val_loss', mode='min', patience=50)

        cb_list = [es]

        model.compile(loss='mean_squared_error', optimizer=nadam)

        self.compute_operator()
        self.data[:,0]  = (self.data[:,0] - min( self.data[:,0]))/ (max(self.data[:,0]) -min(self.data[:,0]))
        self.val = (self.val - min(self.val))/(max(self.val) -min(self.val))
        history = model.fit(self.data, \
                            self.val, callbacks=cb_list, batch_size=32, epochs=500)
        plt.plot(history.history['loss'], label='train')
        #plt.plot(history.history['val_loss'], label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

        prediction_op= model.predict(self.data)
        prediction_op = np.asarray(prediction_op)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x[n_train + n_valid:], y[n_train + n_valid:], f[n_train + n_valid:], c='b', marker='o')
        # ax.scatter(x[n_train + n_valid:], y[n_train + n_valid:], prediction, c='r', marker='s')
        ax.scatter(self.data[:,0], self.data[:,1], self.val, c='b', marker='o')
        ax.scatter(self.data[:,0], self.data[:,1], prediction_op, c='r', marker='s')
        #ax.plot_surface(self.data[:,0], self.data[:,1], self.val, c='b', marker='o')
        #ax.plot_surface(self.data[:,0], self.data[:,1], prediction_op, c='r', marker='s')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel(r'$\beta$')
        plt.show()
