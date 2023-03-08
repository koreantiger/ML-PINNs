#%% Packages 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

from tensorflow import keras
from tensorflow.keras import layers
from numba import njit
from scipy import interpolate

#%% Functions 

@njit()
def compute_residual(vars, vars_n, Theta, nb, nc):     
    z   = vars
    zn  = vars_n
    rhs = np.zeros(nb*nc)
    
    for j in range(nc): #component j
        i = 0
        Fl=flux_function(z[j+i*nc])
        for i in range(nb): #location i
            F = flux_function(z[j+i*nc])
            rhs[j+i*nc] = z[j+i*nc] - zn[j+i*nc] + Theta*(F-Fl) # residual form
            Fl=F           
    return rhs 

@njit()
def jac_numeric(vars, vars_n, Theta, nb, nc):  
    epsilon = 0.0001
    jac = np.zeros((nb*nc,nb*nc)) 
    rhs = compute_residual(vars, vars_n, Theta, nb, nc) 
    var1=np.copy(vars)
    
    for j in range(nc): #with respect to compoenent j
        for i in range(nb): # block i
            var1[j+i*nc] += epsilon
            jac[:,j+i*nc] = (compute_residual(var1, vars_n, Theta, nb, nc) - rhs) / epsilon
            var1[j+i*nc] -= epsilon
    return jac,rhs

@njit
def flux_function(s):
    return s**2 / (s**2 + (1-s)**2) # convex
          

def implicit_solver(z_ini, z_inj, Theta_ref, nb, nc, NT):   
    xi   = np.linspace(0, 1, nb)
    vars = np.zeros(nc*nb)
    temp = np.zeros((nb,Nt))
    
    for i in range(nc):
        z_    = np.ones(nb)*z_ini[i] 
        z_[0] = z_inj[i]    
        vars[i::nc] = z_ 
    
    nit   = 0                               
    Theta = Theta_ref
    
    for t in range(1,NT): #time iterations 
        vars_n = np.copy(vars)
        for n in range(100):  
            jac,rhs = jac_numeric(vars, vars_n, Theta, nb, nc)
            res = np.linalg.norm(rhs)
            
            if (res<1e-5):
                nit+=n+1
                break
            
            dz    = np.linalg.solve(jac,-rhs)
            vars += dz     
            
            Theta = Theta_ref
        
        temp[:,t]=vars
            
        # print('-------------------------------------------------------------------------------------------------------')
        # matprint(jac)
        print('time step={}, n={}, res={}'.format(t,n,res))    
        # plt.plot(xi,vars[0::nc])  
    return xi, temp

#%% simulation 

components = ['C1','C02','C4','C10']
nc = 1
nb = 500            # number of blocks 
Nt = 1000            # number of time steps 
ut  = 0.5
phi = 0.45
dx = 1/nb
dt = 1/Nt
Theta_ref = (ut/phi) * (dt/dx)

z_ini = np.array([0])
z_inj = np.array([1])

xi, temp = implicit_solver(z_ini, z_inj, Theta_ref, nb, nc, Nt)

#%%

Ntt = np.where(temp[-1,:]>0.01)[0][0]
X = np.zeros((50,Ntt+1))
u = np.linspace(0.001,0.999,50)

plt.figure()
plt.grid()
plt.plot(xi,temp[:,:Ntt])
plt.xlabel('x'); plt.ylabel('u')
plt.show()

for i in range(1,Ntt+1):
    flimsy = interpolate.interp1d(temp[:,i],xi,fill_value='extrapolate')
    X[:,i] = flimsy(u)
    if i%100 == 0: 
        plt.figure(1)
        plt.plot(X[:,i],u,'.k')
        plt.plot(xi,temp[:,i],'.r')
        # plt.show()
plt.figure(1)
plt.plot(X[:,Ntt],u,'.k')
plt.plot(xi,temp[:,Ntt],'.r')
plt.xlabel('x'); plt.ylabel('u')
plt.show()

temppp = temp[:,:Ntt+1]
        
np.save('PIML_data', temppp) # u(x,t)
np.save('PIML_data_x_ut', X) # x(u,t)

