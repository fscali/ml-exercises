#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib import cm;
from mpl_toolkits.mplot3d import Axes3D;
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def computeCost(X,Y,theta):
    m=len(Y);
    h=np.dot(X,theta);
    #J=((1.0)/(2*m))* ((h[:,0]-Y)**2).sum(); #need to extract the first column  from h (even if it has only one column)
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
