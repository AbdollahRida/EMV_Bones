import pylab
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import os
os.chdir("C:\Users\abdol\Documents\Ecole Polytechnique\MAP 311\Projet TC")

def normale(mu, sigma2, n):     #renvoie un echantillon de loi normale via la méthode Box-Müller
    
    u1 = np.random.uniform(0, 1, n)
    
    R = [ 0 for i in range(n)]                                  
    for i in range(n):
        tmp = -2 * pylab.math.log(u1[i])
        R[i] = pylab.math.sqrt(tmp)
        
    u2 = np.random.uniform(0, 1, n)
    
    O = [ 0 for i in range(n)]                                  
    
    X = [ 0 for i in range(n)]                                  
    for i in range(n):                                         
        tmp = 2 * pylab.math.pi * u2[i]                         
        O[i] = tmp                                              
        X[i] = R[i] * pylab.math.cos(O[i]) * pylab.math.sqrt(sigma2) + mu
    
    return X


def echantillon( p, mu_1, sigma2_1, mu_2, sigma2_2, n):    #renvoie un echantillon du mélange
    
    x1 = normale(mu_1, sigma2_1, n)
    
    x2 = normale(mu_2, sigma2_2, n)
    
    u = sps.bernoulli.rvs(p, size=n)
    print(u)
    
    x = [ 0 for i in range(n)]
    for i in range(n):
        x[i] = u[i] * x1[i] + (1 - u[i]) * x2[i]
        
    return x

densiteOs=open("densitesOs.txt",'r')
liste=densiteOs.read()
liste=liste.replace("\n","")
liste=liste.replace("  ","   ")
mots=liste.split("   ")

def p(theta,x): #calcule p_i
    return (1-theta[0])*densite_normale(theta[3],theta[4],float(x))/(theta[0]*densite_normale(theta[1],theta[2],x)+(1-theta[0])*densite_normale(theta[3],theta[4],float(x)))