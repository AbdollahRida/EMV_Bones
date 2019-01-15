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
    print(u)
    
    x = [ 0 for i in range(n)]
    for i in range(n):
        x[i] = u[i] * x1[i] + (1 - u[i]) * x2[i]
        
    return x

#Test de l'algorithme

mu_1 = 1.619  #Moyennes
mu_2 = 1.741

sigma2_1 = 0.065 ** 2  #Variances
sigma2_2 = 0.071 ** 2

a = 51.4/100     #Proportions
b = 1.0 - a

pas =1000  #nombre de points

plt.close("all")

E = np.random.randn(int(1e5))  #Data qui marche bien

x = np.linspace(1, 2, pas)
g_x = echantillon( a, mu_1, sigma2_1, mu_2, sigma2_2, pas)  #Test pour une loi normale centrée réduite
f1_x = sps.norm.pdf(x, mu_1, pylab.math.sqrt(sigma2_1))
h1_x = normale( mu_1, sigma2_1, pas)
f2_x = sps.norm.pdf(x, mu_2, pylab.math.sqrt(sigma2_2))
h2_x = normale( mu_2, sigma2_2, pas)
melange = a * f1_x + b * f2_x
plt.plot(x, f1_x, "r", label = "Simulation de loi gaussienne du groupe 1")
#plt.hist(h1_x, bins=50, normed=1, label="Data1")
plt.plot(x, f2_x, "g", label = "Simulation de loi gaussienne du groupe 2")
#plt.hist(h2_x, bins=50, normed=1, label="Data2")

t = [ 0 for i in range(pas)]

for i in range(pas):
    t[i] = a * h1_x[i] + b * h2_x[i]

plt.hist(g_x, bins=20, normed=1, label='test')
plt.plot(x, melange, "b", label = "Simulation de loi du mélange")
#plt.hist(g_x, bins=100, normed=1, label="Data")
#plt.hist(E, bins=50, normed=1, label="Data")  #Pour comparer 
plt.legend(loc="best")
plt.show()
