import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
from numpy import inf

def char_f(s0, v0, theta, k, sigma, r, rho, tao, phi):
    
    a_1 = r*1j*phi*tao
    a_2 = k*theta/sigma**2
    m = np.sqrt((rho*sigma*1j*phi)**2 + sigma**2*(1j*phi+phi**2))
    n = (rho*sigma*1j*phi - k - m)/(rho*sigma*1j*phi - k + m)
    a_3 = -(rho*sigma*1j*phi - k - m)*tao - 2 * np.log((1-n*math.exp(m*tao)/(1-n)))
    first_term = a_1 + a_2*a_3
    second_term = (math.exp(m*tao) - 1)*(rho*sigma*1j*phi - k - m) / (sigma**2 * (1 - n*math.exp(tao)))
    
    return math.exp(first_term+second_term*v0+1j*phi*np.log(s0*math.exp(r*tao)))
  
  char_f(1, 0.16, 0.5, 0.1, 0.2, 0.12, -0.7, 2, 0.1)
  
# Original call price code, returns errors

def cost_f(s0, v0, K, tao, theta, k, sigma, r, rho):
    
    charac_f = lambda phi: char_f(s0, v0, theta, k, sigma, r, rho, tao, phi)
    i1 = np.real((np.exp(-1j * phi * np.log(K)) * charac_f(1j*phi+1)) / (1j * phi * charac_f(1)))
    i2 = np.real((np.exp(phi * np.log(K)) * charac_f(1j*phi)) / (-phi))
    i1_int = quad(i1,-np.inf,+np.inf)
    i2_int = quad(i2,-np.inf,+np.inf)
    p1 = lambda phi: 1/2 + i1_int[0]/np.pi
    p2 = lambda phi: 1/2 + i2_int[0]/np.pi
    cost = s0*p1 - K*np.exp(-r*tao)*p2
    
#Updated call price code

def cost_f(s0, v0, K, tao, theta, k, sigma, r, rho):
    
    charac_f = lambda phi: char_f(s0, v0, theta, k, sigma, r, rho, tao, phi)
    i1 = lambda phi: np.real((np.exp(-1j * phi * np.log(K)) * charac_f(1j*phi+1)) / (1j * phi * charac_f(1)))
    i2 = lambda phi: np.real((np.exp(-1j * phi * np.log(K)) * charac_f(1j*phi)) / (1j*phi))
    i1_int1 = quad(i1,-30,0)
    i1_int2 = quad(i1,0,30)
    i2_int2 = quad(i2,-30,0)
    i2_int2 = quad(i2,0,30)
    
    i1_int = i1_int1 + i1_int2
    i2_int = i2_int2 + i2_int2
    
    p1 = 1/2 + i1_int[0]/np.pi
    p2 = 1/2 + i2_int[0]/np.pi
    cost = s0*p1 - K*np.exp(-r*tao)*p2
    
    print(cost)
    
    plt.scatter(rho,cost,color="b")
    plt.scatter(r,cost,color="g")
    plt.scatter(sigma,cost,color="r")
    plt.scatter(tao,cost,color="y")
    plt.scatter(k,cost,color="m")
    plt.scatter(theta,cost,color="c")
    plt.scatter(K,cost,color="k")
    plt.scatter(v0,cost,color="lawngreen")
    plt.scatter(s0,cost,color="pink")
