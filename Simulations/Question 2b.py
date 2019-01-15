import pylab
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


def normale(mu, sigma2, n):
    
    u1 = np.random.uniform(0, 1, n)
    
    R = [ 0 for i in range(n)]                                  #Dans ce bloc on crée R.
    for i in range(n):
        tmp = -2 * pylab.math.log(u1[i])
        R[i] = pylab.math.sqrt(tmp)
        
    u2 = np.random.uniform(0, 1, n)
    
    O = [ 0 for i in range(n)]                                  #ici on crée thêta, vide d'abord
    
    X = [ 0 for i in range(n)]                                  #On crée X dans ce bloc, on initialise
    for i in range(n):                                          #d'abord puis on applique la formule
        tmp = 2 * pylab.math.pi * u2[i]                         #sur le rapport pour trouver ce qu'il
        O[i] = tmp                                              #faut.
        X[i] = R[i] * pylab.math.cos(O[i]) * pylab.math.sqrt(sigma2) + mu
    
    return X


def echantillon( p, mu_1, sigma2_1, mu_2, sigma2_2, n):
    
    x1 = normale(mu_1, sigma2_1, n)
    
    x2 = normale(mu_2, sigma2_2, n)
    
    u = sps.bernoulli.rvs(p, size=n)
    #print(u)
    
    x = [ 0 for i in range(n)]
    for i in range(n):
        x[i] = u[i] * x1[i] + (1 - u[i]) * x2[i]
    #print(x)    
    return x

#Test de l'algorithme

mu_1 = 1.619  #Moyennes
mu_2 = 1.741

sigma2_1 = 0.065 ** 2  #Variances
sigma2_2 = 0.071 ** 2

p = 51.4/100     #Proportions

pas = 500  #nombre de points

plt.close("all")

x = np.linspace(-10, 10, pas)
g_x = echantillon( p, mu_1, sigma2_1, mu_2, sigma2_2, pas)  #Test pour une loi normale centrée réduite

plt.hist(g_x, bins=20, normed=1, label="Data")

plt.legend(loc="best")
plt.show()

