#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

from __future__ import division;
import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib import cm;
from mpl_toolkits.mplot3d import Axes3D;
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def computeCost(X,Y,theta):
    m=len(Y);
    h=np.dot(X,theta);
    J=((1.0)/(2*m))* ((h-Y)**2).sum(); 

    return J;

def gradientDescent(X,Y,theta,alpha,num_iters):
    m=len(Y);
    J_history=np.zeros((num_iters,1));
    for i in range(num_iters):
        h=np.dot(X,theta);
        
        theta=theta -( alpha/m ) * ( np.dot( np.transpose(X), h-Y ) );

        J_history[i] = computeCost(X,Y,theta);
    return (theta,J_history);

def featureNormalize(X):
    X_norm=X;
    mu=np.mean(X,axis=0);
    sigma=np.std(X,axis=0,ddof=1);
    for i in range(np.size(X,0)):
        X_norm[i,:]=(X[i,:]-mu)/sigma; #division is element wise
    return (X_norm,mu,sigma);

def normalEqn(X,Y):
    return np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),Y));

A=np.eye(5); #identity matrix

data = np.genfromtxt('ex1data1.txt', delimiter=',');
X=data[:,0];
Y=data[:,1];
m=len(Y); #number of training examples
X=data[:,0].reshape(m,1);
Y=data[:,1].reshape(m,1);

plt.figure(1);
plt.plot(X,Y,'rx',  markersize=10);
plt.xlabel('Population of city in 10,000s');
plt.ylabel('Profit in $10,000');
plt.draw();

X=np.transpose(np.vstack([ np.ones((m)), data[:,0] ]));
theta=np.zeros((2,1));
iterations=1500;
alpha=0.01;

J=computeCost(X,Y,theta);
print(J);


[theta, J_history] = gradientDescent(X,Y,theta,alpha,iterations);
print("Theta found by gradient descent: %s" % np.array_str(theta));
plt.plot(X[:,1],np.dot(X,theta),'-');
plt.draw();
#plt.show();

predict1=np.dot(np.array([1, 3.5]), theta);
print("For population=35,000 we predict a profit of %f" % (predict1*10000));

predict2=np.dot(np.array([1, 7]), theta);
print("For population=70,000 we predict a profit of %f" % (predict2*10000));

theta0_vals=np.linspace(-10,10,100);
theta1_vals=np.linspace(-1,4,100);

J_vals=np.zeros((len(theta0_vals), len(theta1_vals)));

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        #t=np.transpose(np.array([theta0_vals[i], theta1_vals[j]]));
        t=np.array([theta0_vals[i], theta1_vals[j]]).reshape(2,1);
        J_vals[i,j]=computeCost(X,Y,t);

# Like in Octave, because of the way meshgrids work in the surf command, we need to 
# transpose J_vals before calling surf, or else the axes will be flipped

J_vals=np.transpose(J_vals);

fig=plt.figure();
ax=fig.gca(projection='3d');
[theta0_vals, theta1_vals] = np.meshgrid(theta0_vals, theta1_vals);
surf=ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False);
ax.set_xlabel('theta0');
ax.set_ylabel('theta1');

ax.zaxis.set_major_locator(LinearLocator(10));
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
fig.colorbar(surf, shrink=0.5, aspect=5);


fig2=plt.figure();
plt.contour(theta0_vals,theta1_vals,J_vals, levels=np.logspace(-2,3,20));
plt.xlabel("theta0");
plt.ylabel("theta1");
plt.draw();

plt.plot(theta[0,:], theta[1,:], 'rx', markersize=10, linewidth=2);
plt.show();
raw_input("Press ENTER to exit");



data = np.genfromtxt('ex1data2.txt', delimiter=',');
X=data[:,0:2];
Y=data[:,2];
m=len(Y); #number of training examples
X=data[:,0:2].reshape(m,2);
Y=data[:,2].reshape(m,1);
print(X.shape);
print(Y.shape);
print("First 10 examples from the dataset:");
for i in range(10):
    print("x=[{x0:f} {x1:f}], y={y:f}".format(x0=X[i,0],x1=X[i,1],y=Y[i,0]));

[X, mu, sigma]=featureNormalize(X);


X=np.hstack([ np.ones((m,1)), data[:,0:2] ]);

print("Running gradient descent..");
alpha=0.5;
num_iters=400;
theta=np.zeros((3,1));
[theta,J_history]=gradientDescent(X,Y,theta,alpha,num_iters);
plt.figure();
plt.plot(range(len(J_history)),J_history,'-b',linewidth=2);
plt.xlabel("Number of iterations");
plt.ylabel("Cost J");
plt.show();

print("Theta computed by gradient descent: [{theta_0:f} {theta_1:f} {theta_2:f}]".format(theta_0=theta[0,0],theta_1=theta[1,0], theta_2=theta[2,0]));
price = 0; 
norm_input = np.array([1.0,1650.0,3.0]).reshape(3,1); #note that I set the float type explicitly otherwise next assignments won't work correctly
norm_input[1,0] = (norm_input[1,0]-mu[0])/(sigma[0]) ;
norm_input[2,0] = (norm_input[2,0]-mu[1])/(sigma[1]);
price =  np.dot(np.transpose(theta), norm_input);
print("Predicted price of a 1650 sq-ft, 3 br house (using gradiend descent):  {my_price:f}".format(my_price=price[0,0]));


theta=normalEqn(X,Y);

print("Theta computed by normal equation: [{theta_0:f} {theta_1:f} {theta_2:f}]".format(theta_0=theta[0,0],theta_1=theta[1,0], theta_2=theta[2,0]));


