# -*- coding: utf-8 -*-
"""
@author: knight
"""
import numpy as np
import matplotlib.pyplot as plt

# def loadDataSet(filename):
#     dataMat = [];
#     fr = open(filename)
#     for line in fr.readlines():
#         dataMat.append(line.strip().split())
#     return dataMat
import scipy as sp

x = sp.genfromtxt("ex2x.dat", delimiter="\t")
y = sp.genfromtxt("ex2y.dat", delimiter="\t")

def error(f,x,y):
    return sp.sum((f(x)-y)**2)
a = 0.07
#theta0_vals,theta1_vals = 0,0
w = [theta0_vals,theta1_vals]

plt.xlabel('Age in years')
plt.ylabel('Height in meters')

# fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
plt.plot(x,y,'o',label="Traning")
# f1 = sp.poly1d(fp1)
# fx = sp.linspace(2,8, 1000)
# plt.plot(fx, f1(fx),label="Linear regression",linewidth=2)
# plt.legend()
plt.show()

# J_vals = np.zeros((100,100))
# theta0_vals = np.linspace(-3,3,100)
# theta1_vals = np.linspace(-1,1,100)
# for i in len(theta0_vals):
#     for j in len(theta1_vals):
#         t = [theta0_vals(i); theta1_vals(j)]
#         J_vals(i,j) =
