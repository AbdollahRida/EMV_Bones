import pylab
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


##1a
def normale(mu, sigma2, n):     #renvoie un échantillon de gaussienne simulé par la méthode de Box-Müller. Pour avoir une seule réalisation mettre n = 1
    
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

##2b
def echantillon( p, mu_1, sigma2_1, mu_2, sigma2_2, n):
    
    x1 = normale(mu_1, sigma2_1, n)
    
    x2 = normale(mu_2, sigma2_2, n)
    
    u = sps.bernoulli.rvs(p, size=n)
    
    x = [ 0 for i in range(n)]
    for i in range(n):
        x[i] = u[i] * x1[i] + (1 - u[i]) * x2[i]
   
    return x

def histogrammes():
    plt.close("all")
    
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

    plt.hist(g_x, bins=20, normed=1, label='Echantillon')
    
    plt.plot(x, melange, "b", label = "Simulation de loi du mélange")
    #plt.hist(g_x, bins=100, normed=1, label="Data")
    #plt.hist(E, bins=50, normed=1, label="Data")  #Pour comparer 
    plt.legend(loc="best")
    plt.show()

##3a
import os
os.chdir( 'C:\\Users\\abdol\\Documents\\Ecole Polytechnique\\MAP 311\\Projet TC')
densiteOs = open("densitesOs.txt",'r')
liste = densiteOs.read()
liste = liste.replace("\n","")
liste = liste.replace("  ","   ")
donnee = liste.split("   ") #Création d'une liste à partir du fichier.

                
def p( theta, x):  #calcule pi
     return (1 - theta[0]) * sps.norm.pdf(x, theta[3], theta[4]) / (theta[0] * sps.norm.pdf(x, theta[1], theta[2]) + (1-theta[0]) * sps.norm.pdf(x, theta[3], theta[4]))
 
    
    
def Q( theta, theta1, donnee):  #donnee est la liste des valeurs
    Q = 0
    for x in donnee:
        Q = Q + np.log(theta[0] * sps.norm.pdf(x, theta[1], theta[2]))*(1 - p( theta1, float(x))) + np.log((1-theta[0]) * sps.norm.pdf(x, theta[3], theta[4]))*p( theta1, float(x))
    return Q

def logvraisemblancecorr(theta): #donne le logarithme de la vraissemblance corrigée
    somme=0
    for x in donnee:
        somme=somme+np.log(theta[0]*sps.norm.pdf(theta[1],theta[2],float(x)))*(1-p(theta,float(x)))+np.log((1-theta[0])*sps.norm.pdf(theta[3],theta[4],float(x)))*p(theta,float(x))
    return somme


def argmaxQ(theta1, donnee):
    global mu_1, mu_2, sigma2_1, sigma_2_2
    
    Sp,Sp1,Sp2,Sp3=0,0,0,0
    S=[0,0,0,0,0]
    
    for x in donnee :
        x1= float(x)
        p1 = p(theta1,x1)
        Sp = Sp + p1
        Sp1 = Sp1 + (1-p1)*x1
        Sp2 = Sp2 + (1-p1)
        Sp3 = Sp3 + p1 * x1
                    
    S[0] = 1 - Sp/len(donnee)
    S[1] = Sp1/Sp2
    S[2] = Sp3/Sp
    
    Sp4,Sp5=0,0
    
    for x in donnee:
        x=float(x)
        p1 = p(theta1,x)
        Sp4 = Sp4 + (1-p1) * (x - S[1])**2
        Sp5 = Sp5 + p1 * (x - S[3])**2
    
    S[3] = Sp4/Sp2
    S[4] = Sp5/Sp
     
    return(S)

def recurrence(n, boolean):
    theta0 = [0.4,2,5,8,9]
    
    print(theta0)
    
    X=np.zeros(n)
   
    S=argmaxQ(theta0,donnee)
    
    for i in range(n):
        #q1 = logvraisemblancecorr(S)
        S = argmaxQ(S,donnee)
        print(S)
        #q2 = logvraisemblancecorr(S)
        #X[i] = q2
        """while abs(q2-q1)>0.001:
            q1=logvraisemblancecorr(S)
            S=argmaxQ(S,donnee)
            q2=logvraisemblancecorr(S)"""
    if boolean:
        plt.plot([i for i in range(n)], X) #trace la courbe de la log vraisemblance
        plt.title('Evolution de la log-vraisemblance en fonction du nombre de récursion')
    return S
        
##Paramètres

   
mu_1 = 1.619  #Moyennes
mu_2 = 1.741

sigma2_1 = 0.065 ** 2  #Variances
sigma2_2 = 0.071 ** 2

a = 51.4/100     #Proportions
b = 1.0 - a

pas = 500  #nombre de points

E = np.random.randn(int(1e5))    


