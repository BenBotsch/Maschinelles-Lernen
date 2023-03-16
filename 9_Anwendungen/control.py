#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 21:57:32 2022

@author: ben
"""

import sys
sys.path.append("../Utils")
from plot import plot
import numpy as np
import matplotlib.pyplot as plt
from random import uniform,randint
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize

    
def cost_function(u,w,out,dt):
    """
    Computes the cost function for a given control 
    input, output signal, and time step.

    Parameters:
    -----------
        u : float
            Control input.
        w : float
            Desired output signal.
        out : list
            List containing previous output signals.
        dt : float
            Time step.

    Returns:
    --------
        float : 
            The computed cost value.
    """
    a=[1.,0.125,0.25]
    b=[1.]
    if(len(out)>1):
        ref0 = dt**2/a[2]*(2*a[2]/dt**2-a[1]/dt-a[0])*out[-1]
        ref1 = dt**2/a[2]*(-a[2]/dt**2+a[1]/dt)*out[-2]
        ref2 = dt**2/a[2]*(b[0])*u
        y=ref0+ref1+ref2
        cost = (y-w)**2
        if(u>10 or u<-10):
            cost+=50
        return cost
    else:
        return 0
    
class RealSystem:
    """
    A class representing a real system with a 
    transfer function of the form:
    G(z) = b / (a[0] + a[1]*z^-1 + a[2]*z^-2)
    """
    def __init__(self,
                 a:list,
                 b:list,
                 dt:float=0.1):
        """
        Initializes the RealSystem object.

        Parameters:
        ----------
        a : list
            A list of three coefficients of the denominator 
            polynomial of the transfer function.
        b : list
            A list of coefficients of the numerator 
            polynomial of the transfer function.
        dt : float, optional
            The sampling time in seconds.
        """
        self.out=[]
        self.a = a
        self.b = b
        self.dt = dt
        self.time = []
        self.current_time = 0
        
    def simulate(self,
                 u:float,
                 u_time:float,
                 dt:float=0.1,
                 iterations:int=100):
        """
        Simulates the system for a given input 
        u until u_time.

        Parameters:
        ----------
        u : float
            The input value to the system.
        u_time : float
            The time to stop simulating the system.
        dt : float, optional
            The sampling time in seconds. 
        iterations : int, optional
            The number of iterations to simulate.
        """
        self.reset()
        for i in range(iterations):
            self.current_time+=dt
            self.time.append(self.current_time)
            if(len(self.out)>1):
                if(self.current_time>u_time):
                    self.out.append(self.output(u))
                else:
                    self.out.append(np.ravel(0))
            else:
                self.out.append(np.ravel(0))
                
    def step(self,u:float):
        """
        Simulates one time step of the system for a given input u.

        Parameters:
        ----------
        u : float
            The input value to the system.
        """
        dt=self.dt
        self.current_time+=dt
        self.time.append(self.current_time)
        self.out.append(self.output(u))
        
    def output(self,u:float):
        """
        Computes the output of the system for a given input u.

        Parameters:
        ----------
        u : float
            The input value to the system.

        Returns:
        -------
        np.ndarray
            The output value of the system.
        """
        dt=self.dt
        if(len(self.out)>1):
            ref0 = dt**2/self.a[2]*(2*self.a[2]/dt**2-self.a[1]/dt-self.a[0])*self.out[-1]
            ref1 = dt**2/self.a[2]*(-self.a[2]/dt**2+self.a[1]/dt)*self.out[-2]
            ref2 = dt**2/self.a[2]*(self.b[0])*u
            y=ref0+ref1+ref2
            return np.ravel(y)
        else:
            return np.ravel(0)
    def reset(self):
        self.out=[]
        self.time=[]
        self.current_time=0
                
class Model:
    """
    A class representing a neural network 
    model with the ability to fit data,
    predict output based on input, and 
    return the loss value.
    """
    def __init__(self):
        """
        Initializes a MLPRegressor object with solver, alpha, hidden_layer_sizes, and random_state parameters.
        """
        self.clf = MLPRegressor(solver='sgd', 
                                 alpha=1e-5,
                                 hidden_layer_sizes=(50, 50, 10),
                                 random_state=1)
    def fit(self,X,y):
        """
        Fits the MLPRegressor with the given input data and target values.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.
        """
        self.clf.partial_fit(X, y)
    def predict(self,X):
        """
        Predicts the target values for the given input data using the trained neural network.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        return self.clf.predict(X)
    def loss(self):
        """
        Returns the loss value of the trained neural network.

        Returns:
        --------
        loss : float
            The loss value of the trained neural network.
        """
        return self.clf.loss_
    
class Controller:   
    """
    A class representing a controller used to predict 
    a system's behavior and generate control signals.
    """
    def __init__(self):
        """
        Initializes the Controller object with an 
        MLPRegressor object with specified parameters.
        """
        self.clf = MLPRegressor(solver='sgd', 
                                 alpha=1e-5,
                                 hidden_layer_sizes=(100, 50, 20),
                                 random_state=1)
    def fit(self,X,y):
        """
        Fits the MLPRegressor object to the provided 
        input-output training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data used for training the 
            MLPRegressor object.

        y : array-like, shape (n_samples,)
            The target values used for training the 
            MLPRegressor object.

        Returns:
        --------
        None
        """
        self.clf.partial_fit(X, y)
        
    def predict(self,X):
        """
        Predicts the output of the system based on the input X using the MLPRegressor object.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data for which to predict the output.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted output values corresponding to the input X.
        """
        return self.clf.predict(X)
    def loss(self):
        """
        Returns the loss value of the MLPRegressor object.

        Parameters:
        -----------
        None

        Returns:
        --------
        loss_value : float
            The loss value of the MLPRegressor object.
        """
        return self.clf.loss_
        
    
def train_model(model, system, jumps=150):
    """
    Trains the given model using the 
    provided system data.

    Parameters:
    -----------
        model:  
            An instance of a model class that has 
            'fit' and 'predict' methods implemented.
        system: 
            An instance of a system class that has 
            'reset', 'step', and 'out' properties 
            implemented.
        jumps (int):    
            The number of jumps the system 
            should make during training.

    Returns:
    --------
        None
    """
    n_input=20
    system.reset()
    yp=np.zeros((n_input,))
    y=np.zeros((n_input,))
    y_rand=np.zeros((n_input,))
    X=np.zeros((n_input,n_input))
    X_train=np.zeros((n_input,n_input+1))
    for _ in range(X.shape[1]):
        system.step(u=np.zeros((1,)))
    for i in range(jumps):
        wi = uniform(-3,3)
        steps = randint(80,150)
        for _ in range(steps):
            system.step(u=wi)
            rand = np.random.uniform(low=-0.05, high=0.05, size=(1,X.shape[1]))
            y = np.concatenate((y,np.ravel(system.out[-1])))
            y_rand = np.concatenate((y_rand,np.ravel(system.out[-1]+uniform(-0.1,0.1))))
            X = np.concatenate((X,np.array(system.out[-1-n_input:-1]).reshape(1, -1)+rand))
            X_train = np.concatenate((X_train,np.concatenate((np.array(X[-1,:]).reshape(1, -1),np.array(wi).reshape(1, -1)),axis=1)))
            model.fit(X_train, y)
            yp = np.concatenate((yp,model.predict(np.array(X_train[-1,:]).reshape(1, -1))))
        print('\r', 'Jump:', i, ' Loss:', model.loss(), end='')

    plot(x=[system.time,system.time],
         y=[y_rand,yp],
         xlabel="t",
         ylabel="y(t)",
         ylim=(-5,7),
         xlim=(0,70),
         legend=["reales System","Modell"],
         legend_loc="upper right")

def test_model(model, system, u_range, i_range):
    n_input=20
    system.reset()
    yp=np.zeros((n_input,))
    y=np.zeros((n_input,))
    y_rand=np.zeros((n_input,))
    X=np.zeros((n_input,n_input))
    X_test=np.zeros((n_input,n_input+1))
    w=np.zeros((n_input,))
    for _ in range(X.shape[1]):
        system.step(u=np.zeros((1,)))
    for u, i_index in zip(u_range, range(len(i_range))):
        for i in range(i_range[i_index]):
            system.step(u=u)
            rand = np.random.uniform(low=-0.01, high=0.01, size=(1,X.shape[1]))
            w = np.concatenate( (w,np.ravel(u)) )
            y = np.concatenate((y,np.ravel(system.out[-1])))
            y_rand = np.concatenate((y_rand,np.ravel(system.out[-1]+uniform(-0.1,0.1))))
            X = np.concatenate((X[-1,1:].reshape(1, -1),np.array(yp[-1]).reshape(1, -1)),axis=1)+rand
            X_test = np.concatenate((X,np.array(u).reshape(1, -1)),axis=1)
            yp = np.concatenate((yp,model.predict(X_test)))

    plot(x=[system.time,system.time],
         y=[w,yp],
         xlabel="t",
         ylabel="y(t)",
         ylim=(-0.1,2.5),
         figsize_x=4.5,
         #legend=["reales System","Modell"],
         legend_loc="upper right")
    
def train_controller(model, system, controller, jumps=200):
    """
    Trains a controller for a given system using a model.

    Parameters:
    -----------
        model :object
            A model object that defines 
            the system's behavior.
        system : object
            A system object that defines 
            the system's dynamics.
        controller : object
            A controller object that will 
            be trained to control the system.
        jumps : int 
            The number of iterations to 
            train the controller.

    Returns:
    --------
    None
    """
    n_input=20
    system.reset()
    X=np.zeros((n_input,n_input))
    X_train=np.zeros((n_input,n_input+1))
    u_opt=np.zeros((n_input,))
    ur = np.zeros((1,))
    w=np.zeros((n_input,))
    for _ in range(X.shape[1]):
        system.step(u=ur)
    for i in range(jumps):
        wi = uniform(-3,3)
        steps = randint(80,150)
        for _ in range(steps):
            res = minimize(cost_function, 0.01, args=(wi,system.out,system.dt), method='BFGS', tol=1e-6,options={"maxiter": 500})
            ur = res.x
            u_opt = np.concatenate( (u_opt,np.ravel(ur)) )
            rand = np.random.uniform(low=-0.001, high=0.001, size=(X.shape[1],))
            w = np.concatenate( (w,np.ravel(wi)) )
            X = np.concatenate( (X,np.array(system.out[-n_input:]).reshape(1, -1)+rand) )
            X_train = np.concatenate( 
                (X_train,
                  np.concatenate((X[-1,:].reshape(1, -1), np.array(wi).reshape(1, -1)),axis=1)))
            controller.fit(X_train, u_opt)
            system.step(u=ur)
        print('\r', 'Jump:', i, ' Loss:', controller.loss(), end='')

    plt.plot(w)
    plt.plot(system.out)
    plt.show()
    plot(x=[system.time,system.time],
         y=[w,system.out],
         xlabel="t",
         ylabel="y(t)",
         ylim=(-5,7),
         #xlim=(0,70),
         legend=["Führungsgröße","reales System"],
         legend_loc="upper right")
    return controller

def test_controller(model, system, controller, u_range, i_range):
    n_input=20
    system.reset()
    y=np.zeros((n_input,))
    X=np.zeros((n_input,n_input))
    X_test=np.zeros((n_input,n_input+1))
    ur = np.zeros((1,))
    w=np.zeros((n_input,))
    for _ in range(X.shape[1]):
        system.step(u=ur)
    for wi, i_index in zip(u_range, range(len(i_range))):
        for _ in range(i_range[i_index]):
            rand = np.random.uniform(low=-0.001, high=0.001, size=(X.shape[1],))
            w = np.concatenate( (w,np.ravel(wi)) )
            y = np.concatenate((y,np.ravel(system.out[-1])))
            X = np.array(system.out[-n_input:]).reshape(1, -1)+rand
            X_test = np.concatenate((X, np.array(wi).reshape(1, -1)),axis=1)
            ur = controller.predict(X_test)
            system.step(u=ur)
            
    plot(x=[system.time,system.time],
         y=[w,system.out],
         xlabel="t",
         ylabel="y(t)",
         ylim=(-0.1,2.5),
         figsize_x=4.5,
         #legend=["Führungsgröße","reales System"],
         legend_loc="upper right")
    
if __name__ == "__main__":
    
    model = Model()
    system = RealSystem([1.,0.125,0.25],[1.])
    controller = Controller()
    #system.simulate(1, 1, 0.01, 2000)

    #train model
    train_model(model, system)
    
    # test model
    u_range=[1.5,1]
    i_range=[200,100]
    test_model(model, system, u_range, i_range)
    
    # train controller
    controller = train_controller(model, system, controller)
    
    # test controller
    u_range=[1.5,0.5,1.0]
    i_range=[100,100,100]
    test_controller(model, system, controller, u_range, i_range)

    
    
    
    