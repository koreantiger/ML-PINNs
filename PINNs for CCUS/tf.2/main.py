import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

from scipy.interpolate import griddata
from tensorflow import keras
from keras import backend as K
from pyDOE import lhs

"""
Implimentation of PIML for BL displacement as described in Fucks and Tchelepi (2020)
"""

#%% IMPORT AND PLOT DATA 

# plant seed for random but predictable sampling 
np.random.seed(1234)
tf.random.set_seed(1234)

Nu = 300   # number of sample points for MSE-term, boundary and initial conditions (max. nb+Nt) 
Nr = 10000 # number of collocation points for the residual term, internal 
Nl = 0     # number of points for lebeseque term

data_u_xt   = np.load('PIML_data.npy')       # loading u(x,t), [nb x Nt] 
data_x_ut   = np.load('PIML_data_x_ut.npy')  # loading x(u,t) = f^-1(u(x,t)), [50 x Nt]

nb =  data_u_xt.shape[0]   # number of blocks 
Nt =  data_u_xt.shape[1]   # number of time steps 
t  = np.linspace(0,1,Nt)   # time vector
# t = np.linspace(0,0.735,Nt) 
# tt = np.linspace(0,1,736)
x  = np.linspace(0,1,nb)   # position vector    
u  = np.linspace(0.001,0.999,data_x_ut.shape[0]) # uniform vector for u

plt.figure(figsize=(10,5))
plt.grid()
plt.plot(x, data_u_xt[:,0], '-', label='u(x,0)')
# plt.plot(data_x_ut[:,0], u, '.-', label='x(u,0)')
plt.plot(x, data_u_xt[:,-1], '-x', label='u(x,%d)'%Nt)
plt.plot(data_x_ut[:,-1], u, '.-', label='x(u,%d)'%Nt)
plt.legend()
plt.xlabel('xd')
plt.ylabel('u')
plt.show()

#%% CONSTRUCT TRAINING DATA SETS 

# transpose u(x,t) and x(u,t)
Exact_u_xt = data_u_xt.T 
Exact_x_ut = data_x_ut.T

# meshgrid for u(x,t) and x(u,t)
X, T      = np.meshgrid(x,t) # [Nt x nb]
U, Travis = np.meshgrid(u,t) # [Nt x 50]

# vectorized form u(x,t) and attributes 
XT_temp = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # attributes: [<X>,<T>]
u_temp  = Exact_u_xt.flatten()[:,None] # 'labels': [< u(x,t) >]

# vectorized form of x(u,t) and attributes 
UT_temp = np.hstack((Exact_x_ut.flatten()[:,None], Travis.flatten()[:,None])) # attributes: [<U>,<T>]
x_temp  = U.flatten()[:,None] # [< x(u,t) >]   

# Domain bounds for normalisation (not strictly necessary)
lb = XT_temp.min(0)
ub = XT_temp.max(0)       

# construct data set containing initail and boundary conditions
    # Ɐx and t=0 -> u(x,t) = 0
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) 
uu1 = Exact_u_xt[0:1,:].T

    # x = 0 and t>0 -> u(x,t) = 1      
xx2 = np.hstack((X[1:,0:1], T[1:,0:1]))
uu2 = Exact_u_xt[1:,0:1]

    # complete set of boundary and initial points 
XT_u_train = np.vstack([xx1, xx2])
u_train    = np.vstack([uu1, uu2])

plt.figure(dpi=100)
plt.grid()
plt.title('Complete set of boundary and intial points')
c = plt.scatter(XT_u_train[:,0], XT_u_train[:,1], c=u_train, marker = 'x')
plt.colorbar(c)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

    # randomly sample complete set of boundary and intial points Nu times  
idx        = np.random.choice(Nt+nb-1, Nu, replace=False) # replace = False ensures that a point is not sampled more than once 
XT_u_train = XT_u_train[idx,:] 
u_train    = u_train[idx,:]

plt.figure(dpi=100)
plt.grid()
plt.title('Sampled boundary and intial points for training')
c = plt.scatter(XT_u_train[:,0], XT_u_train[:,1], c=u_train, marker = 'x')
plt.colorbar(c)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

# randomly sample collocation points for residual term
XT_r_train = lb + (ub-lb)*lhs(2, Nr)             # Latin Hypercube Sampling of collocation points 
XT_r_train = np.vstack((XT_r_train, XT_u_train)) # Add sampled boundary points to collocation points 

# randomly sample points for lebesque term
idx        = np.random.choice(UT_temp.shape[0], Nl, replace=False)
UT_x_train = UT_temp[idx]
x_train    = x_temp[idx]

# add lebesgue points to MSE term 
u_train    = np.vstack([u_train,x_train])
XT_u_train = np.vstack([XT_u_train,UT_x_train]) 

# plot u(x,t) and x(u,t)
    #u(x,t)
plt.figure(figsize=(10,5))
plt.title('u(x,t)')
plt.plot(XT_u_train[:,1:2],XT_u_train[:,0:1], 'xr', label='boundary points (labelled)')
plt.scatter(XT_r_train[:,1:2],XT_r_train[:,0:1], s=1, label='collocation points (unlabelled)')
h=plt.imshow(Exact_u_xt.T, interpolation='nearest', cmap='plasma', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')

plt.xlim([-0.1,1.1]); plt.ylim([-0.1,1.1])
plt.colorbar(h)
plt.legend()
plt.xlabel('t')
plt.ylabel('x')
plt.show()

#%% NN MODEL DESCRIPTION 

inputs = keras.Input(shape=(2,), name="nameless")
x = keras.layers.Dense(20, activation = "tanh")(inputs)
x = keras.layers.Dense(20, activation = "tanh")(x)
x = keras.layers.Dense(20, activation = "tanh")(x)
x = keras.layers.Dense(20, activation = "tanh")(x)
x = keras.layers.Dense(20, activation = "tanh")(x)
x = keras.layers.Dense(20, activation = "tanh")(x)
x = keras.layers.Dense(20, activation = "tanh")(x)
x = keras.layers.Dense(20, activation = "tanh")(x)
out = keras.layers.Dense(1, activation = 'linear', name="predictions_1")(x)
model = keras.Model(inputs=inputs, outputs=out) # create model 
model.summary() # print model description 

#%% TRAINING LOOP 

loss_fn    = keras.losses.MeanSquaredError() # declare loss function
epochs     = 10000                            # number of epochs/iterations 
batch_size = XT_u_train.shape[0]             # batch size 
lr         = 1e-5                            # learning rate 
optimizer  = keras.optimizers.Adam(lr)        # optimizer 
# optimizer  = keras.optimizers.Adam(lr)

training_set  = tf.data.Dataset.from_tensor_slices((XT_u_train, u_train)).batch(batch_size)
XT_r_train_tf = tf.Variable(XT_r_train, dtype=tf.float32)

loss_1 = [] # list to append loss values to 
loss_2 = []
loss   = [] 
for epoch in range(epochs):
    with tf.GradientTape(persistent=True) as tape:
        # persistent = True --> allows you to call tape more than once 
        for step, (XT_u_train_tf, u_train_tf) in enumerate(training_set):
            # labelled term 
            u1  = model(XT_u_train_tf)
            L1  = loss_fn(u_train_tf, u1)
            
            # residual term
            with tf.GradientTape(persistent=True) as gg:
                u2  = model(XT_r_train_tf)
                f   = u2**2/(u2**2+(1-u2)**2)
            
            u_t = gg.gradient(u2, XT_r_train_tf)[:,1] # du/dt
            f_x = gg.gradient(f , XT_r_train_tf)[:,0] # df/dx
                
            L2  = tf.reduce_mean(tf.square(u_t + 0.55*f_x))
            
            # total loss 
            L  = L1 + L2
            
            loss_1.append(L1)
            loss_2.append(L2)
            loss.append(L)
    
    # compute gradients with respect to trainable weights 
    grads = tape.gradient(L, model.trainable_weights) # dL/dW
    
    # apply gradient update to weights 
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # print loss statistics per epoch 
    print("Training loss at epoch %d: L1 + L2 = %.4e + %.4e = %.4e" % (epoch, float(L1), float(L2), float(L)))
    del tape
    del gg 
    
    if epoch % 100 == 0:
    #     u_pred = model.predict(XT_temp)
    #     U_pred = griddata(XT_temp, u_pred.flatten(), (X, T), method='linear')
    #     sample = np.arange(0,736,100)
        
    #     plt.figure()
    #     plt.title('epoch %d'%epoch)
    #     plt.grid()
    #     plt.plot(Exact_u_xt[0::100].T,'-k',label='exact')
    #     plt.plot(U_pred[0::100,:].T,'--r',label='prediction')
    #     # plt.xlim([0,1])
    #     # plt.ylim([0,1])
    #     plt.show()
        
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.grid()
        plt.title('loss 1')
        plt.plot(loss_1, label='loss_1')
        
        plt.subplot(122)
        plt.grid()
        plt.title('loss 2')
        plt.plot(loss_2, label='loss_2')
        plt.show()
        
#%%

u_pred = model.predict(XT_temp)
U_pred = griddata(XT_temp, u_pred.flatten(), (X, T), method='linear')
sample = np.arange(0,736,100)

plt.figure()
plt.title('epoch %d'%epoch)
plt.grid()
plt.plot(Exact_u_xt[0::100].T,'-k',label='exact')
plt.plot(U_pred[0::100,:].T,'--r',label='prediction')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.show()

plt.figure()
plt.grid()
plt.plot(loss_1, label='loss_1')
plt.plot(loss_2, label='loss_2')
plt.show()

    
        
    
