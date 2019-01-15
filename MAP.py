##I
from random import gauss
import numpy as np
import matplotlib.pyplot as plt

##I.1
def normale(mu,sigma2): #simule une loi gaussienne
    return np.random.normal(mu, sigma2)

##I.2
mu_1=1.619
mu_2=1.741
sigma_1=0.065
sigma_2=0.071
proportion=0.514

def echantillon(n): #renvoie un tableau des n réalisations de la loi demandée
    tab=[0 for i in range(n)]
    for i in range(n):
        tab[i]=proportion*normale(mu_1,sigma_1**2)+(1-proportion)*normale(mu_2,sigma_2**2)
    return tab

def histogrammes(): #trace les histogrammes demandés en I.2.b
    
    #n=100
    plt.subplot(211)
    Y=echantillon(100)
    plt.hist(Y,bins=50,normed=1,label="n=100",color='lightblue')
    plt.legend(loc='best')
    plt.show
    
    #n=1000
    plt.subplot(212)
    Y=echantillon(1000)
    plt.hist(Y,bins=50,normed=1,label="n=1000",alpha=0.4,color='red')
    plt.legend(loc='best')
    plt.show
    
    
def densite_normale(mu,sigma,x): #densité d'une loi normale
    return np.exp(-((x-mu)/sigma)**2/2)/(2*np.pi*sigma**2)**(1/2)

def melange(): #trace les trois gaussiennes demandées en I.2.c
    X=np.linspace(1,2,1000)
    Y1=densite_normale(mu_1,sigma_1,X)
    Y2=densite_normale(mu_2,sigma_2,X)
    Y3=proportion*densite_normale(mu_1,sigma_1,X)+(1-proportion)*densite_normale(mu_2,sigma_2,X)
    plt.plot(X,Y1)
    plt.plot(X,Y2)
    plt.plot(X,Y3)
    plt.legend(loc='best')
    plt.show

##II.3.ab
import numpy as np
import os
os.chdir("C:/Users/Matthieu RP/OneDrive/Bureau ASUS")

densiteOs=open("densitesOs.txt",'r')
liste=densiteOs.read()
liste=liste.replace("\n","")
liste=liste.replace("  ","   ")
mots=liste.split("   ") #rendre le fihier texte de données utilisable

def p(theta,x): #calcule p_i
    return (1-theta[0])*densite_normale(theta[3],theta[4],float(x))/(theta[0]*densite_normale(theta[1],theta[2],x)+(1-theta[0])*densite_normale(theta[3],theta[4],float(x)))

def Q(theta, thetaprime): #donne Q(theta,theta')
    Q=0
    for x in valeurs:
        X=float(x) #on convertit le string x en float X
        Q=Q+np.log(densite_normale(theta[1],theta[2],X)*(1-p(thetaprime,X)))+np.log(densite_normale(theta[3],theta[4],X))*p(thetaprime,X)
    return Q

def maximisation(theta,valeurs): #calcule theta(i+1) à partir de theta(i)

    #calcul de m0 et m1 les deux moyennes de theta, chacune rapport de deux sommes
    somme1,somme2,somme3,somme4=0,0,0,0
    
    for x in valeurs :
        X=float(x)
        somme1=somme1+(1-p(theta,X))*X
        somme2=somme2+(1-p(theta,X))
        somme3=somme3+p(theta,X)*X
        somme4=somme4+p(theta,X)
        
        

    m0=somme1/somme2
    m1=somme3/somme4
    
    #calcul de v0 et v1 les deux variances de theta, chacune rapport de deux sommes
    somme5,somme6,somme7,somme8=0,0,0,0
    for x in valeurs:
        X=float(x)
        somme5=somme5+(1-p(theta,X))*(X-m0)**2
        somme6=somme6+1-p(theta,X)
        somme7=somme7+p(theta,X)*(X-m1)**2
        somme8=somme8+p(theta,X)
    
    v0=somme5/somme6
    v1=somme7/somme8
    
    return [theta[0],m0,v0,m1,v1] #renvoie theta(i+1)
    
def logvraisemblancecorr(theta): #donne le logarithme de la vraissemblance corrigée
    somme=0
    for x in mots:
        somme=somme+np.log(theta[0]*densite_normale(theta[1],theta[2],float(x)))*(1-p(theta,float(x)))+np.log((1-theta[0])*densite_normale(theta[3],theta[4],float(x)))*p(theta,float(x))
    return somme

def recurrence(n,boolean): #recurrence demandée en II.3.a
    theta0=[0.5,0.7,1,1,1] #Etape 0
    X=np.zeros(n)
    for i in range(n):
        theta0=maximisation(theta0,mots)
        X[i]=logvraisemblancecorr(theta0) #crée un tableau des valeurs de la log-vraise
    if boolean:
        plt.plot(X) #trace la courbe de la log vraisemblance
        plt.title('Evolution de la log-vraisemblance enfonction du nombre de récursion')
    return theta0 #renvoie theta0 maximisé n fois.

##II.4

def LGN(valeurs,k): #calcule la moyenne des k premiers Z de la liste
    n=len(valeurs)
    somme=0
    for i in range(k):
        somme=somme+float(valeurs[i])
    return somme/k

def comparaison(k): #affiche le graphe de la différence entre les deux méthodes
    tab=np.zeros(k)
    for i in range(1,k):
        tab[i]=abs(recurrence(5,False)[1]-LGN(mots,i))
    plt.plot(tab)
    plt.title('Evolution de la difference absolue entre les deux méthodes suivant le nombre de données')
