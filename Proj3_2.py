#!/usr/bin/env python
# coding: utf-8

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

LW=5 # linewidth
MS=10 # markersize

g = 9.81 # [m/s2] freefall accerleration 
qc = 0.1
Dt = 0.01 # time step
Nsims = 500 # number of timesteps


# ## Extended Kalman Filter

# In[35]:


def prediction_step(Phi, A, mean, cov, noise_cov):
    """Prediction Step: Propagate uncertainty for one time step

    X_{k+1} = A X_{k} + xi, xi sim mathcal{N}{0, noise_cov}
    X_{k} sim mathcal{N}(mean, cov)

    Inputs
    ------
    Phi: callable, non-linear system
    A: callable, first order term of Taylor series (Gradient of Phi)
    mean: (d, ) prior mean
    cov: (d, d) prior covariance, symmetric positive definite
    noise_cov: (d, d) process noise covariance, symmetric positive definite

    Returns
    -------
    pred_mean: (d, ) predicted mean
    pred_cov: (d, d) predicted covariance
    """
    
    pred_mean = Phi(mean)
    pred_cov = np.dot(A(mean), np.dot(cov, A(mean).T)) + noise_cov
    
    return pred_mean, pred_cov


# In[36]:


def update_step(data, h, H, mean, cov, noise_cov):
    """Update Step
    
    Inputs
    ------
    data: (d,)
    h: callable, measurement function
    H: callable first order term of Taylor series (Gradient of h)
    mean: (d) prior mean
    cov: (d, d) prior covariance
    noise_cov : (d, d) noise covariance matrix
    
    Returns
    -------
    update_mean: (d, ) updated mean
    update_cov: (d, d) updated covariance
    
    """

    mu = h(mean)
    H_mean = H(mean)
    U = np.dot(cov, H_mean.T)
    S = np.dot(H_mean, np.dot(cov, H_mean.T)) + noise_cov
    update_mean = mean[:, np.newaxis] + np.dot(U, np.linalg.inv(S)) * (data[0] - mu[0])
    update_mean = np.reshape(update_mean, 2)
    update_cov = cov - np.dot(U, np.linalg.solve(S, U.T))
    
    return update_mean, update_cov


# In[37]:


def extended_kalman_filter(data, Phi, A, proc_cov, h, H, meas_cov, prior_mean, prior_cov):
    """The Extended Kalman filter
    
    Inputs
    ------
    data: (N, m), N is the number of time steps, m is the size of the observations
    Phi: callable, non-linear system
    A: callable, first order term of Taylor series (Gradient of Phi)
    proc_cov: (d,d), process noise covariance
    h: callable, measurement function
    H: callable, first order term of Taylor series (Gradient of h)
    meas_cov: (m, m), measurement noise covariance
    prior_mean: (d, ) prior mean
    prior_cov: (d, d), prior_covariance
    
    Return
    ------
    mean_store: (N+1, d): posterior means (first row is the prior)
    cov_store: (d, d, N+1): posterior covariances (first block is the prior)    
    """
    
    d = prior_mean.shape[0]
    N = data.shape[0]
    m = data.shape[1]
    
    mean_store = np.zeros((N+1, d))
    mean_store[0, :] = np.copy(prior_mean)
    cov_store = np.zeros((d, d, N+1))
    cov_store[:, :, 0] = np.copy(prior_cov)
    
    #Loop over all data
    for ii in range(N):
        # Prediction
        pred_mean, pred_cov = prediction_step(Phi, A, mean_store[ii, :], cov_store[:, :, ii], proc_cov)

        # Update
        if ii % delta != 0:
            mean_store[ii+1, :] = pred_mean
            cov_store[:, :, ii+1] = pred_cov
        elif ii % delta == 0:
            mean_store[ii+1, :], cov_store[:, :, ii+1] = update_step(data[ii, :], h, H, pred_mean, pred_cov, meas_cov)
        
    return mean_store, cov_store


# ## Model
# \begin{equation}
#     \begin{bmatrix}
#         x_1^{k+1}\\
#         x_2^{k+1}
#     \end{bmatrix}
#     =
#     \begin{bmatrix}
#         x_1^k + x_2^k \Delta t \\
#         x_2^k - g\sin{(x_1^k)} \Delta t
#     \end{bmatrix}
#     +
#     q^k
#     = \Phi(\mathbf{X}) + q^k
# \end{equation}
# where $q^k \sim N(0, Q)$
# \begin{equation}
#     Q = 
#     \begin{bmatrix}
#         \frac{q^c \Delta t^3}{3} & \frac{q^c \Delta t^2}{2} \\
#         \frac{q^c \Delta t^2}{2} & q^c \Delta t
#     \end{bmatrix}
# \end{equation}
# with $q^c = 0.1$ and $\Delta t = 0.01$
# 
# Initial condition $(x_1^0, x_2^0) = (1.5, 0)$

# In[38]:


def Phi(X):
    '''
    Dynamic model
    ---------------------
    inputs:
    X: (d), states
    ---------------------
    outputs:
    X_next: (N, d), next states
    '''
    
    X_next = X.copy()
    X_next[0] = X[0] + X[1] * Dt
    X_next[1] = X[1] - g * np.sin(X[0]) * Dt
    
    return X_next


# #### Compute $\nabla \Phi$
# \begin{equation}
#     \Phi(\mathbf{X}) = 
#     \begin{bmatrix}
#         f_1(\mathbf{X}) \\
#         f_2(\mathbf{X})
#     \end{bmatrix}
#     =
#     \begin{bmatrix}
#         x_1^k + x_2^k \Delta t \\
#         x_2^k - g\sin{(x_1^k)} \Delta t
#     \end{bmatrix}
# \end{equation}
# 
# \begin{equation}
#     A_k = \nabla \Phi(\mathbf{X}) = 
#     \begin{bmatrix}
#         \frac{\partial f_1}{\partial x_1^k} & \frac{\partial f_1}{\partial x_2^k} \\
#         \frac{\partial f_2}{\partial x_1^k} & \frac{\partial f_2}{\partial x_2^k}
#     \end{bmatrix}
#     =
#     \begin{bmatrix}
#         1 & \Delta t\\
#         -g\cos{(x_1^k)} \Delta t & 1
#     \end{bmatrix}
# \end{equation}

# In[39]:


def grad_Phi(X):
    '''
    Gradient of Dynamic model
    ---------------------
    inputs:
    X: (d), states
    ---------------------
    outputs:
    grad_X (d, d), gradient (Jacobian) of Phi
    '''
    d = X.shape[0]
    grad_X = np.ones((d, d))
    grad_X[0, 1] = Dt
    grad_X[1, 0] = -g * np.cos(X[0]) * Dt
    
    return grad_X


# #### compute $h(\mathbf{X})$
# 
# \begin{equation}
#     y^{\delta k} = \sin{(x_1^{\delta k})} + r^{\delta k}
# \end{equation}
# where $r^{\delta k } \sim N(0, R)$

# In[40]:


def h(X):
    '''
    Measurement function
    ---------------------
    inputs:
    X: (d), states
    ---------------------
    outputs:
    y: (1,), measurement outputs
    '''
    y = np.copy(X)
    y[0] = np.sin(X[0])
    y[1] = 0
    
    return y 


# #### compute $ \nabla h(\mathbf{X})$
# 
# \begin{equation}
#     H_k = \nabla h(\mathbf{X}) = 
#     \begin{bmatrix}
#         \cos{(x_1^{\delta k})} & 0
#     \end{bmatrix}
# \end{equation}

# In[41]:


def grad_h(X):
    '''
    Gradient of Measurement function
    ---------------------
    inputs:
    X: (d), states
    ---------------------
    outputs:
    y: (1, 1), gradient (Jacobian) of measurement function
    '''
    
    grad_h = np.array([[np.cos(X[0]), 0]])
    
    return grad_h


# ### Set up problem

# In[42]:


def generate_truth(N, x0, h, Phi, noise_std=1e-2):
    """Generate the truth for the second-order system"""
    
    xout = np.zeros((N+1, 2)) 
    
    Phi_X = Phi(xout[0, :])
    h_X = h(xout[0, :])
    yout = np.zeros((N, 2))
    xout[0, :] = x0
    
    for ii in range(N):
        xout[ii+1, :] = Phi(xout[ii, :])
        if ii%delta == 0:
            yout[ii, :] = h(xout[ii+1, :]) + noise_std * np.random.randn(1)
            yout[ii, 1] = 0
        else:
            yout[ii, :] = np.copy(yout[ii-1, :])
    return xout, yout


# Initialize the prior means and covariances

# In[43]:


def get_std(cov):
    """Get square root of diagonals (standard deviations) from covariances """
    
    d, d, N = cov.shape
    std_devs = np.zeros((N, d))
    for ii in range(N):
        std_devs[ii, :] = np.sqrt(np.diag(cov[:, :, ii]))
    return std_devs


# In[44]:


ExKF_mean_post = np.zeros((16, Nsims+1, 2))
ExKF_cov_post = np.zeros((16, 2, 2, Nsims+1))

plt.figure(figsize=(16,16))

i = 1
t = np.linspace(0, Dt*Nsims, Nsims+1)

for delta in [5, 10, 20, 40]:
    for R in [1, 0.1, 0.01, 0.001]:
        ax = plt.subplot(4, 4, i)
        
        x0 = np.array([1.5, 0])
        xout, yout = generate_truth(Nsims, x0, h, Phi, noise_std=np.sqrt(R))
        
        prior_mean = np.copy(x0) # prior mean
        prior_cov = np.eye(2)  # prior covariance
        proc_cov = np.array([[qc * (Dt)**3 / 3, qc * (Dt)**2 / 2], [qc * (Dt)**2 / 2, qc * (Dt)]])
        meas_cov = np.copy(R)
        
        start = time.time()
        meanpost, covpost = extended_kalman_filter(yout, Phi, grad_Phi, proc_cov, h, grad_h, meas_cov, prior_mean, prior_cov)
        end = time.time()
        print("elapse time {:.2f}".format(end - start))
        ExKF_mean_post[i-1, :, :] = meanpost
        ExKF_cov_post[i-1, :, :, :] = covpost
        
        std_devs = get_std(covpost) #extract the standard deviations at all the states
        
        ax.plot(t, xout[:, 0],'--', color='red', label='state0truth')
        ax.plot(t, xout[:, 1],'--', color='blue', label='state1truth')

        ax.plot(t, meanpost[:, 0], color='red', label='state0')
        ax.plot(t, meanpost[:, 1], color='blue', label='state1') 
        ax.fill_between(t, meanpost[:, 0] - 2 * std_devs[:, 0],  meanpost[:, 0] + 2*std_devs[:, 0],
                        color='red', alpha=0.3)
        ax.fill_between(t, meanpost[:, 1] - 2 * std_devs[:, 1],  meanpost[:, 1] + 2*std_devs[:, 1],
                        color='blue', alpha=0.3)
        
        MSE0 = np.sum((meanpost[:, 0] - xout[:, 0])**2)/Nsims##
        MSE1 = np.sum((meanpost[:, 1] - xout[:, 1])**2)/Nsims##
        tex = "MSE0: {:.2f}\nMSE1: {:.2f}".format(MSE0, MSE1) 
        ax.text(0.7, 0.1, tex, transform=ax.transAxes)
        
        for ii in range(1, Nsims+1, delta):
            plt.plot(t[ii], yout[ii, 0], 'ko', alpha=0.4)
            
        if R==1 and delta==5:##
            ax.legend(loc=2)##
        
        if delta==40:
            ax.set_xlabel('Time', fontsize=14)
        
        if R==1:
            ax.set_ylabel('State Estimate',fontsize=14)
            
        ax.set_title('R: ' + str(R) + '; delta: ' + str(delta))
        
        i += 1
        
plt.show()


# ## Unscented Kalman Filter

# In[12]:


def unscented_points(mean, cov, alg='chol', alpha=1, beta=0, kappa=0):
    """Generate unscented points"""    
    dim = cov.shape[0]
    lam = alpha*alpha*(dim + kappa) - dim
    if alg == "chol":
        L = np.linalg.cholesky(cov)
    elif alg == "svd":
        u, s, v = np.linalg.svd(cov)
        L = np.dot(u, np.sqrt(np.diag(s)))
    pts = np.zeros((2*dim+1, 2))
    pts[0, :] = mean
    for ii in range(1, dim+1):        
        pts[ii, :] = mean + np.sqrt(dim + lam)*L[:, ii-1]        
        pts[ii+dim,:] = mean - np.sqrt(dim + lam)*L[:, ii-1]

    W0m = lam / (dim + lam)
    W0C = lam / (dim + lam) + (1 - alpha*alpha + beta)
    Wim = 1/2 / (dim + lam)
    Wic = 1/2 / (dim + lam)
    return pts, (W0m, Wim, W0C, Wic)


# In[13]:


def unscented_prediction_step(Phi, mean, cov, noise_cov, UP, W):
    """Prediction Step (UKF): Propagate uncertainty for one time step using unscented Kalman Filter

    X_{k+1} = A X_{k} + xi, xi sim mathcal{N}{0, noise_cov}
    X_{k} sim mathcal{N}(mean, cov)

    Inputs
    ------
    Phi: callable, non-linear system
    mean: (d, ) prior mean
    cov: (d, d) prior covariance, symmetric positive definite
    noise_cov: (d, d) process noise covariance, symmetric positive definite
    UP: unscented points
    W: weights

    Returns
    -------
    pred_mean: (d, ) predicted mean
    pred_cov: (d, d) predicted covariance
    UP_Phi: Phi of Unscented Points
    """
    dim = cov.shape[0]
    UP_Phi = np.zeros((2*dim + 1, 2)) # Phi of sigma points 
    pred_mean = np.zeros((2))
    
    pred_mean = W[0] * Phi(UP[0, :])
    for ii in range(1, 2*dim + 1):        
        UP_Phi[ii, :] = Phi(UP[ii, :])
        pred_mean += W[1] * UP_Phi[ii, :]
    
    delta = UP_Phi - pred_mean
    
    pred_cov = W[2] * np.dot(delta[0, :][:, np.newaxis], delta[0, :][np.newaxis, :])
    for ii in range(1, 2*dim + 1):
        pred_cov += W[3] * np.dot(delta[ii, :][:, np.newaxis], delta[ii, :][np.newaxis, :])
    pred_cov += noise_cov
    
    return pred_mean, pred_cov, UP_Phi


# In[14]:


def unscented_update_step(data, h, mean, cov, noise_cov, UP, W, UP_Phi):
    """Update Step (UKF)
    
    Inputs
    ------
    data: (d,)
    h: callable, measurement function
    mean: (d) prior mean
    cov: (d, d) prior covariance
    noise_cov : (d, d) noise covariance matrix
    UP: unscented points
    W: weights
    UP_Phi: Phi of Unscented Points
    
    Returns
    -------
    update_mean: (d, ) updated mean
    update_cov: (d, d) updated covariance
    
    """
    dim = cov.shape[0]
    
    
    
    for ii in range(0, 2*dim + 1):
        UP_Phi_h[ii, :] = h(UP_Phi[ii, :])
        
    mu = W[0] * UP_Phi_h[0, :]
    for ii in range(1, 2*dim + 1):
        mu += W[1] * UP_Phi_h[ii, :]
        
    delta = UP_Phi_h[:, 0] - mu[0]
    
    S = W[2] * np.dot(delta[0], delta[0])
    for ii in range(1, 2*dim + 1):
        S += W[3] * np.dot(delta[ii], delta[ii])
    S += noise_cov
    
    deltaX = UP_Phi - mean
    
    U = W[2] * np.dot(deltaX[0, :][:, np.newaxis], delta[0])
    for ii in range(1, 2*dim + 1):
        U += W[3] * np.dot(deltaX[ii, :][:, np.newaxis], delta[ii])
    
    update_mean = mean[:, np.newaxis] + U / S * (data[0] - mu[0])
    update_mean = np.reshape(update_mean, 2)
    update_cov = cov - np.dot(U/S, U.T)
    
    return update_mean, update_cov


# In[15]:


def unscented_kalman_filter(data, Phi, proc_cov, h, meas_cov, prior_mean, prior_cov):
    """The Unscented Kalman filter
    
    Inputs
    ------
    data: (N, m), N is the number of time steps, m is the size of the observations
    Phi: callable, non-linear system
    proc_cov: (d,d), process noise covariance
    h: callable, measurement function
    meas_cov: (m, m), measurement noise covariance
    prior_mean: (d, ) prior mean
    prior_cov: (d, d), prior_covariance
    
    Return
    ------
    mean_store: (N+1, d): posterior means (first row is the prior)
    cov_store: (d, d, N+1): posterior covariances (first block is the prior)    
    """
    
    d = prior_mean.shape[0]
    N = data.shape[0]
    m = data.shape[1]
    
    mean_store = np.zeros((N+1, d))
    mean_store[0, :] = np.copy(prior_mean)
    cov_store = np.zeros((d, d, N+1))
    cov_store[:, :, 0] = np.copy(prior_cov)
    
    #Loop over all data
    for ii in range(N):
        
        # Unscented points & parameters
        
        UP, W = unscented_points(mean_store[ii, :], cov_store[:, :, ii])
        
        # Prediction
        pred_mean, pred_cov, UP_Phi = unscented_prediction_step(Phi, mean_store[ii, :], cov_store[:, :, ii], proc_cov, UP, W)

        # Update
        if ii % delta != 0:
            mean_store[ii+1, :] = pred_mean
            cov_store[:, :, ii+1] = pred_cov
        elif ii % delta == 0:
            mean_store[ii+1, :], cov_store[:, :, ii+1] = unscented_update_step(data[ii, :], h, pred_mean, pred_cov, meas_cov, UP, W, UP_Phi)
    return mean_store, cov_store


# In[16]:


UKF_mean_post = np.zeros((16, Nsims+1, 2))
UKF_cov_post = np.zeros((16, 2, 2, Nsims+1))

plt.figure(figsize=(16,16))

i = 1
t = np.linspace(0, Dt*Nsims, Nsims+1)

for delta in [5, 10, 20, 40]:
    for R in [1, 0.1, 0.01, 0.001]:
        ax = plt.subplot(4, 4, i)
        
        x0 = np.array([1.5, 0])
        xout, yout = generate_truth(Nsims, x0, h, Phi, noise_std=np.sqrt(R))
        
        prior_mean = np.copy(x0) # prior mean
        prior_cov = np.eye(2)  # prior covariance
        proc_cov = np.array([[qc * (Dt)**3 / 3, qc * (Dt)**2 / 2], [qc * (Dt)**2 / 2, qc * (Dt)]])
        meas_cov = np.copy(R)
        
        start = time.time()
        meanpost, covpost = unscented_kalman_filter(yout, Phi, proc_cov, h, meas_cov, prior_mean, prior_cov)
        end = time.time()
        print("elapse time {:.2f}".format(end - start))
        
        UKF_mean_post[i-1, :, :] = meanpost
        UKF_cov_post[i-1, :, :, :] = covpost
        
        std_devs = get_std(covpost) #extract the standard deviations at all the states
        
        ax.plot(t, xout[:, 0],'--', color='red', label='state0truth')
        ax.plot(t, xout[:, 1],'--', color='blue', label='state1truth')

        ax.plot(t, meanpost[:, 0], color='red', label='state0')
        ax.plot(t, meanpost[:, 1], color='blue', label='state1') 
        ax.fill_between(t, meanpost[:, 0] - 2 * std_devs[:, 0],  meanpost[:, 0] + 2*std_devs[:, 0],
                        color='red', alpha=0.3)
        ax.fill_between(t, meanpost[:, 1] - 2 * std_devs[:, 1],  meanpost[:, 1] + 2*std_devs[:, 1],
                        color='blue', alpha=0.3)
        
        MSE0 = np.sum((meanpost[:, 0] - xout[:, 0])**2)/Nsims##
        MSE1 = np.sum((meanpost[:, 1] - xout[:, 1])**2)/Nsims##
        tex = "MSE0: {:.2f}\nMSE1: {:.2f}".format(MSE0, MSE1) 
        ax.text(0.7, 0.1, tex, transform=ax.transAxes)
        
        for ii in range(1, Nsims+1, delta):
            plt.plot(t[ii], yout[ii, 0], 'ko', alpha=0.4)
            
        if R==1 and delta==5:##
            ax.legend(loc=2)##
        
        if delta==40:
            ax.set_xlabel('Time', fontsize=14)
        
        if R==1:
            ax.set_ylabel('State Estimate',fontsize=14)
            
        ax.set_title('R: ' + str(R) + '; delta: ' + str(delta))
        
        i += 1
        
plt.show()


# ## Gauss-hermite Kalman Filter

# In[25]:


def gh_oned(num_pts=2):
    """Gauss-hermite quadrature in 1D"""
    A = np.zeros((num_pts, num_pts))
    for ii in range(num_pts):
        #print("ii ", ii, ii==0, ii==(order-1))
        row = ii+1
        if ii == 0:
            A[ii, ii+1] = np.sqrt(row)
            A[ii+1, ii] = np.sqrt(row)
        elif ii == (num_pts-1):
            A[ii-1, ii] = np.sqrt(ii)
        else:
            A[ii, ii+1] = np.sqrt(row)
            A[ii+1, ii] = np.sqrt(row)
    pts, evec = np.linalg.eig(A)
    devec = np.dot(evec.T, evec)
    wts = evec[0,:]**2
    
    return pts, wts

def tensorize(nodes):
    """Tensorize nodes to obtain twod"""
    n1d = nodes.shape[0]
    twodnodes = np.zeros((n1d*n1d, 2))
    ind = 0
    for ii in range(n1d):
        for jj in range(n1d):
            twodnodes[ind, :] = np.array([nodes[ii], nodes[jj]])
            ind +=1
    return twodnodes

def gauss_hermite(dim, num_pts=2):
    """Gauss-hermite quadrature in 2D"""
    assert dim == 2, "Tensorize only implemented for dim=2"
    pts, weights = gh_oned(num_pts)
    ptsT = tensorize(pts)
    weightsT = tensorize(weights)
    weightsT = np.prod(weightsT, axis=1)
    return ptsT, weightsT

def rotate_points(points, mean, cov, alg="chol"):
    """Rotating points from standard gaussian to target Gaussian"""
    if alg == "chol":
        L = np.linalg.cholesky(cov)
    elif alg == "svd":
        u, s, v = np.linalg.svd(cov)
        L = np.dot(u, np.sqrt(np.diag(s)))
        

    new_points = np.zeros(points.shape)
    for ii in range(points.shape[0]):
        new_points[ii, :] = mean + np.dot(L, points[ii,:].T)
    return new_points


# In[26]:


def gauss_hermite_prediction_step(Phi, mean, cov, noise_cov, GHP, W):
    """Prediction Step (GHKF): Propagate uncertainty for one time step using gauss hermite Kalman Filter

    X_{k+1} = A X_{k} + xi, xi sim mathcal{N}{0, noise_cov}
    X_{k} sim mathcal{N}(mean, cov)

    Inputs
    ------
    Phi: callable, non-linear system
    mean: (d, ) prior mean
    cov: (d, d) prior covariance, symmetric positive definite
    noise_cov: (d, d) process noise covariance, symmetric positive definite
    GHP: ghu points points
    W: weights

    Returns
    -------
    pred_mean: (d, ) predicted mean
    pred_cov: (d, d) predicted covariance
    UP_Phi: Phi of Unscented Points
    """
    
    n_GHP = GHP.shape[0]
    dim = cov.shape[0]
    GHP_Phi = np.zeros((GHP.shape)) # Phi of sigma points 
    pred_mean = np.zeros((2))
    
    for ii in range(n_GHP):        
        GHP_Phi[ii, :] = Phi(GHP[ii, :])
        pred_mean += W[ii] * GHP_Phi[ii, :]
    
    delta = GHP_Phi - pred_mean
    
    pred_cov = np.zeros((dim, dim))
    for ii in range(n_GHP):
        pred_cov += W[ii] * np.dot(GHP_Phi[ii, :][:, np.newaxis], GHP_Phi[ii, :][np.newaxis, :])
    pred_cov += noise_cov
    pred_cov -= np.dot(pred_mean[:, np.newaxis], pred_mean[np.newaxis, :])
    
    return pred_mean, pred_cov, GHP_Phi


# In[27]:


def gauss_hermite_update_step(data, h, mean, cov, noise_cov, GHP, W, GHP_Phi):
    """Update Step (GHKF)
    
    Inputs
    ------
    data: (d,)
    h: callable, measurement function
    mean: (d) prior mean
    cov: (d, d) prior covariance
    noise_cov : (d, d) noise covariance matrix
    UP: unscented points
    W: weights
    GHP_Phi: Phi of gauss hermite Points
    
    Returns
    -------
    update_mean: (d, ) updated mean
    update_cov: (d, d) updated covariance
    
    """
    dim = cov.shape[0]
    n_GHP = GHP_Phi.shape[0]
    
    GHP_h = np.zeros(GHP.shape)
    
    for ii in range(n_GHP):
        GHP_h[ii, :] = h(GHP[ii, :])
        
    GHP_h = GHP_h[:, 0]
    
    mu = 0
    for ii in range(n_GHP):
        mu += W[ii] * GHP_h[ii]
    
    #delta = GHP_Phi_h[:, 0] - mu[0]
    
    S = 0
    for ii in range(n_GHP):
        S += W[ii] * GHP_h[ii] * GHP_h[ii]
    S += noise_cov
    S -= mu * mu
    
    # deltaX = GHP_Phi - mean
    
    GHP_mean = np.mean(GHP, axis=0)
    
    U = np.zeros((dim, 1))
    for ii in range(n_GHP):
        U += W[ii] * GHP[ii, :][:, np.newaxis] * GHP_h[ii]
    U -= GHP_mean[:, np.newaxis] * mu
    
    update_mean = mean[:, np.newaxis] + U / S * (data[0] - mu)
    update_mean = np.reshape(update_mean, 2)
    update_cov = cov - np.dot(U/S, U.T)
    
    return update_mean, update_cov


# In[28]:


def gauss_hermite_kalman_filter(data, Phi, proc_cov, h, meas_cov, prior_mean, prior_cov):
    """The gauss hermite Kalman filter
    
    Inputs
    ------
    data: (N, m), N is the number of time steps, m is the size of the observations
    Phi: callable, non-linear system
    proc_cov: (d,d), process noise covariance
    h: callable, measurement function
    meas_cov: (m, m), measurement noise covariance
    prior_mean: (d, ) prior mean
    prior_cov: (d, d), prior_covariance
    
    Return
    ------
    mean_store: (N+1, d): posterior means (first row is the prior)
    cov_store: (d, d, N+1): posterior covariances (first block is the prior)    
    """
    
    d = prior_mean.shape[0]
    N = data.shape[0]
    m = data.shape[1]
    
    mean_store = np.zeros((N+1, d))
    mean_store[0, :] = np.copy(prior_mean)
    cov_store = np.zeros((d, d, N+1))
    cov_store[:, :, 0] = np.copy(prior_cov)
    
    GHPs, W = gauss_hermite(2, num_pts=3)
    #Loop over all data
    for ii in range(N):
        
        # GH points & parameters
        GHP = rotate_points(GHPs, mean_store[ii, :], cov_store[:, :, ii], alg="chol")
        
        # Prediction
        pred_mean, pred_cov, GHP_Phi = gauss_hermite_prediction_step(Phi, mean_store[ii, :], cov_store[:, :, ii], proc_cov, GHP, W)
        
        # Update
        if ii % delta != 0:
            mean_store[ii+1, :] = pred_mean
            cov_store[:, :, ii+1] = pred_cov
        elif ii % delta == 0:
             # GH points & parameters
            GHP = rotate_points(GHPs, pred_mean, pred_cov, alg="chol")
            mean_store[ii+1, :], cov_store[:, :, ii+1] = gauss_hermite_update_step(data[ii, :], h, pred_mean, pred_cov, meas_cov, GHP, W, GHP_Phi)
    return mean_store, cov_store


# In[29]:


GHKF_mean_post = np.zeros((16, Nsims+1, 2))
GHKF_cov_post = np.zeros((16, 2, 2, Nsims+1))

plt.figure(figsize=(16,16))

i = 1
t = np.linspace(0, Dt*Nsims, Nsims+1)

for delta in [5, 10, 20, 40]:
    for R in [1, 0.1, 0.01, 0.001]:
        ax = plt.subplot(4, 4, i)
        
        x0 = np.array([1.5, 0])
        xout, yout = generate_truth(Nsims, x0, h, Phi, noise_std=np.sqrt(R))
        
        prior_mean = np.copy(x0) # prior mean
        prior_cov = np.eye(2)  # prior covariance
        proc_cov = np.array([[qc * (Dt)**3 / 3, qc * (Dt)**2 / 2], [qc * (Dt)**2 / 2, qc * (Dt)]])
        meas_cov = np.copy(R)
        
        start = time.time()
        meanpost, covpost = gauss_hermite_kalman_filter(yout, Phi, proc_cov, h, meas_cov, prior_mean, prior_cov)
        end = time.time()
        print("elapse time {:.2f}".format(end - start))
        
        GHKF_mean_post[i-1, :, :] = meanpost
        GHKF_cov_post[i-1, :, :, :] = covpost
        
        std_devs = get_std(covpost) #extract the standard deviations at all the states
        
        ax.plot(t, xout[:, 0],'--', color='red', label='state0truth')
        ax.plot(t, xout[:, 1],'--', color='blue', label='state1truth')

        ax.plot(t, meanpost[:, 0], color='red', label='state0')
        ax.plot(t, meanpost[:, 1], color='blue', label='state1') 
        ax.fill_between(t, meanpost[:, 0] - 2 * std_devs[:, 0],  meanpost[:, 0] + 2*std_devs[:, 0],
                        color='red', alpha=0.3)
        ax.fill_between(t, meanpost[:, 1] - 2 * std_devs[:, 1],  meanpost[:, 1] + 2*std_devs[:, 1],
                        color='blue', alpha=0.3)
        
        MSE0 = np.sum((meanpost[:, 0] - xout[:, 0])**2)/Nsims
        MSE1 = np.sum((meanpost[:, 1] - xout[:, 1])**2)/Nsims
        tex = "MSE0: {:.2f}\nMSE1: {:.2f}".format(MSE0, MSE1) 
        ax.text(0.7, 0.1, tex, transform=ax.transAxes)

        
        for ii in range(1, Nsims+1, delta):
            plt.plot(t[ii], yout[ii, 0], 'ko', alpha=0.4)
        
        if R==1 and delta==5:
            ax.legend(loc=2)
        
        if delta==40:
            ax.set_xlabel('Time', fontsize=14)
        
        if R==1:
            ax.set_ylabel('State Estimate',fontsize=14)
            
        ax.set_title('R: ' + str(R) + '; delta: ' + str(delta))
        
        i += 1
        
plt.show()


# In[32]:


def gauss_hermite_kalman_filter_order5(data, Phi, proc_cov, h, meas_cov, prior_mean, prior_cov):
    """The gauss hermite Kalman filter
    
    Inputs
    ------
    data: (N, m), N is the number of time steps, m is the size of the observations
    Phi: callable, non-linear system
    proc_cov: (d,d), process noise covariance
    h: callable, measurement function
    meas_cov: (m, m), measurement noise covariance
    prior_mean: (d, ) prior mean
    prior_cov: (d, d), prior_covariance
    
    Return
    ------
    mean_store: (N+1, d): posterior means (first row is the prior)
    cov_store: (d, d, N+1): posterior covariances (first block is the prior)    
    """
    
    d = prior_mean.shape[0]
    N = data.shape[0]
    m = data.shape[1]
    
    mean_store = np.zeros((N+1, d))
    mean_store[0, :] = np.copy(prior_mean)
    cov_store = np.zeros((d, d, N+1))
    cov_store[:, :, 0] = np.copy(prior_cov)
    
    GHPs, W = gauss_hermite(2, num_pts=5)
    #Loop over all data
    for ii in range(N):
        
        # GH points & parameters
        GHP = rotate_points(GHPs, mean_store[ii, :], cov_store[:, :, ii], alg="chol")
        
        # Prediction
        pred_mean, pred_cov, GHP_Phi = gauss_hermite_prediction_step(Phi, mean_store[ii, :], cov_store[:, :, ii], proc_cov, GHP, W)
        
        # Update
        if ii % delta != 0:
            mean_store[ii+1, :] = pred_mean
            cov_store[:, :, ii+1] = pred_cov
        elif ii % delta == 0:
             # GH points & parameters
            GHP = rotate_points(GHPs, pred_mean, pred_cov, alg="chol")
            mean_store[ii+1, :], cov_store[:, :, ii+1] = gauss_hermite_update_step(data[ii, :], h, pred_mean, pred_cov, meas_cov, GHP, W, GHP_Phi)
    return mean_store, cov_store


# In[33]:


GHKF5_mean_post = np.zeros((16, Nsims+1, 2))
GHKF5_cov_post = np.zeros((16, 2, 2, Nsims+1))

plt.figure(figsize=(16,16))

i = 1
t = np.linspace(0, Dt*Nsims, Nsims+1)

for delta in [5, 10, 20, 40]:
    for R in [1, 0.1, 0.01, 0.001]:
        ax = plt.subplot(4, 4, i)
        
        x0 = np.array([1.5, 0])
        xout, yout = generate_truth(Nsims, x0, h, Phi, noise_std=np.sqrt(R))
        
        prior_mean = np.copy(x0) # prior mean
        prior_cov = np.eye(2)  # prior covariance
        proc_cov = np.array([[qc * (Dt)**3 / 3, qc * (Dt)**2 / 2], [qc * (Dt)**2 / 2, qc * (Dt)]])
        meas_cov = np.copy(R)
        
        start = time.time()
        meanpost, covpost = gauss_hermite_kalman_filter_order5(yout, Phi, proc_cov, h, meas_cov, prior_mean, prior_cov)
        end = time.time()
        print("elapse time {:.2f}".format(end - start))
        GHKF5_mean_post[i-1, :, :] = meanpost
        GHKF5_cov_post[i-1, :, :, :] = covpost
        
        std_devs = get_std(covpost) #extract the standard deviations at all the states
        
        ax.plot(t, xout[:, 0],'--', color='red', label='state0truth')
        ax.plot(t, xout[:, 1],'--', color='blue', label='state1truth')

        ax.plot(t, meanpost[:, 0], color='red', label='state0')
        ax.plot(t, meanpost[:, 1], color='blue', label='state1') 
        ax.fill_between(t, meanpost[:, 0] - 2 * std_devs[:, 0],  meanpost[:, 0] + 2*std_devs[:, 0],
                        color='red', alpha=0.3)
        ax.fill_between(t, meanpost[:, 1] - 2 * std_devs[:, 1],  meanpost[:, 1] + 2*std_devs[:, 1],
                        color='blue', alpha=0.3)
        
        MSE0 = np.sum((meanpost[:, 0] - xout[:, 0])**2)/Nsims
        MSE1 = np.sum((meanpost[:, 1] - xout[:, 1])**2)/Nsims
        tex = "MSE0: {:.2f}\nMSE1: {:.2f}".format(MSE0, MSE1) 
        ax.text(0.7, 0.1, tex, transform=ax.transAxes)

        
        for ii in range(1, Nsims+1, delta):
            plt.plot(t[ii], yout[ii, 0], 'ko', alpha=0.4)
        
        if R==1 and delta==5:
            ax.legend(loc=2)
        
        if delta==40:
            ax.set_xlabel('Time', fontsize=14)
        
        if R==1:
            ax.set_ylabel('State Estimate',fontsize=14)
            
        ax.set_title('R: ' + str(R) + '; delta: ' + str(delta))
        
        i += 1
        
plt.show()


# In[24]:


plt.figure(figsize=(16,16))

i = 1
t = np.linspace(0, Dt*Nsims, Nsims+1)

for delta in [5, 10, 20, 40]:
    for R in [1, 0.1, 0.01, 0.001]:
        ax = plt.subplot(4, 4, i)
        
        x0 = np.array([1.5, 0])
        xout, yout = generate_truth(Nsims, x0, h, Phi, noise_std=np.sqrt(R))
        
        ax.plot(t, xout[:, 0] - xout[:, 0],'--', color='k', label='state0_truth')
        ax.plot(t, (ExKF_mean_post[i-1, :, 0] - xout[:, 0])**2, color='r', label='state0_ExKF')
        ax.plot(t, (UKF_mean_post[i-1, :, 0] - xout[:, 0])**2, color='g', label='state0_UKF')
        ax.plot(t, (GHKF_mean_post[i-1, :, 0] - xout[:, 0])**2, color='b', label='state0_GHKF_order3')
        ax.plot(t, (GHKF5_mean_post[i-1, :, 0] - xout[:, 0])**2, color='purple', label='state0_GHKF_order5')
        
        if R==1 and delta==5:
            ax.legend(loc=2)
        
        if delta==40:
            ax.set_xlabel('Time', fontsize=14)
        
        if R==1:
            ax.set_ylabel(r'$(\hat{X} - X_{true})^2$',fontsize=14)
            
        ax.set_title('R: ' + str(R) + '; delta: ' + str(delta))
        ax.set_ylim(-0.5, 2)
        
        i += 1


# In[ ]:




