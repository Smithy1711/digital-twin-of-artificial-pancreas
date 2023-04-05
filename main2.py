import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pygad
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# Time points
T = 1000
dt = 1.0
time = np.arange(0, T, dt)

# steady-state values
Ib = 10
Gb = 100 

l = 600 #cm, l is the length of the small intestine from the jejunum to the ilium, generally 600-700cm
u = 5 #cm/s, u is the average transit time of the small intestine, generally 0.1-10cm/s
        #velocity will depend on factors such as the contractile force of the intestine, the viscosity of the contents of the intestine, and the diameter of the intestine

tau = l/u

# Initial conditions
S0 = 75
J0 = 0.0
L0 = 0.0

G0 = 25
I0 = 12

Gprod0 = 2

def Stomach(kjs):
    S = np.zeros_like(time)
    S[0] = S0
    for i in range(1, len(time)):
        dSdt = -kjs*S[i-1]
        S[i] = S[i-1] + dSdt*dt
    return S

def Jejunum(kjs, kgj, kjl):
    J = np.zeros_like(time)
    J[0] = J0
    S = Stomach(kjs)
    for i in range(1, len(time)):
        dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]
        J[i] = J[i-1] + dJdt*dt
    return J

def Ileum(kjs, kgj, kjl, kgl):
    L = np.zeros_like(time)
    L[0] = L0
    J = Jejunum(kjs, kgj, kjl)
    for i in range(1, len(time)):
        # Calculate phi at current time step
        if time[i] < tau:
            phi = 0
        else:
            phi = J[np.floor(i-tau).astype(int)]
        # Calculate derivative at current time step
        dLdt = kjl * phi - kgl * L[i-1]
        L[i] = L[i-1] + dLdt*dt
    return L

def Glucose_Insulin(kjs, kgj, kjl, kgl, kxi, kxg, kxgi, n, klambda, k2, x):
    #G_ = np.zeros_like(time)
    #G_[0] = G0
    I = np.zeros_like(time)
    I[0] = I0
    G = np.zeros_like(time)
    G[0] = G0
    Gprod = np.zeros_like(time)
    Gprod[0] = Gprod0
    J = Jejunum(kjs, kgj, kjl)
    L = Ileum(kjs, kgj, kjl, kgl)
    S = Stomach(kjs)


    for i in range(1, len(time)):
        Gprod[i] = (klambda * (Gb-G[i-1])) / (k2 + (Gb - x)) + Gprod[0] 
        
        #klambda - mM2 min−1 Kinetic constant for hepatic glucose release rate

        
        #G_[i] = G[i-1] + np.sum(fgj * (kgj * J[i-1] + kgl * L[i-1])) not needed in a person with type 1 diabetes

        dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod[i-1] + n*(kgj*J[i-1] + kgl*L[i-1])
        
        # when a value >0 is given for G0 then dGdt will give the natural loss of glucose.
        # also making there be no bolus then J and L are 0. also I is 0 as there is no administration of insuline
        # therefore dGdt is just -kxg*G[i-1]

        G[i] = G[i-1] + dGdt*dt # when dgdt*dt becomes <0 find value of dgdt. this gives natural loss of glucose in the blood.


        # dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib) #look at each part of the equation in the paper
        # beta and gamma are not needed in a diabetic person because they are not producing insulin. beta and gamma are parameters for half saturation and acceleration of the insulin production 
        # insulin can take up to 30 mins before it starts to work. 1 to 3 hours for the peak activity. https://www.diabetes.co.uk/insulin/insulin-actions-and-durations.html
        # can replace the acceleration with a delay? also may need to account for basal insulin or "left over insulin" in the body
        # dIdt = kxi * Ib * ((1) / (((Gb/G_[i-1])) + 1) - I[i-1]/Ib)

        if i < 30:
            dIdt = 0
        else:
            dIdt = kxi * (- I[i-1])
        I[i] = I[i-1] + dIdt*dt

    return S, G, I, J, L, Gprod

def main():
    kjs = 0.034
    kxi = 0.025
    kgj = 0.067
    kjl = 0.0087
    kgl = 0.0002
    kxg = 0.078
    kxgi = 0.028
    n = 0.01
    klambda = 0.02
    k2 = 22
    x = 16

    initialParameters = [kjs, kxi, kgj, kjl, kgl, kxg, kxgi, n, klambda, k2, x]

    S, G, I, J, L, Gprod = Glucose_Insulin(kjs, kgj, kjl, kgl, kxi, kxg, kxgi, n, klambda, k2, x)
    return S, G, I, J, L, Gprod, initialParameters

S, G, I, J, L, Gprod, IP = main()

np.savetxt('dataS.csv', S, delimiter=',')
np.savetxt('dataG.csv', G, delimiter=',')
np.savetxt('dataI.csv', I, delimiter=',')
np.savetxt('dataJ.csv', J, delimiter=',')
np.savetxt('dataL.csv', L, delimiter=',')
np.savetxt('dataGprod.csv', Gprod, delimiter=',')


#parameter fitting

def objectiveS(x, data):
    S = Stomach(x)
    S_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(S_)):
        S_[i] = S[T[i]-1]
    error = np.sum(np.square(S_ - data)) #mean squared error
    
    return error


def objectiveJ(x, kjs, data):
    kgj, kjl = x
    J = Jejunum(kjs, kgj, kjl)
    J_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(J_)):
        J_[i] = J[T[i]-1]
    error = np.sum(np.square(J_ - data)) #mean squared error
    
    return error

def objectiveL(x, kjs, kgj, kjl, data):
    kgl = x
    L = Ileum(kjs, kgj, kjl, kgl)
    L_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(L_)):
        L_[i] = L[T[i]-1]
    error = np.sum(np.square(L_ - data)) #mean squared error
    
    return error


def objectiveGprod(x, data, G):
    klambda, k2, x1 = x
    g = gprod(klambda, k2, x1, G)
    g_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(g_)):
        g_[i] = g[T[i]-1]
    error = np.sum(np.square(g_ - data)) #mean squared error

    return error

#supplementary function for Gprod parameter fitting
def gprod(klambda, k2, x1, G):
    Gprod = np.zeros_like(time)
    Gprod[0] = Gprod0
    for i in range(1, len(time)):
        Gprod[i] = (klambda * (Gb-G[i-1])) / (k2 + (Gb - x1)) + Gprod[0]
    return Gprod

def objectiveGI(x, kjs, kgj, kjl, kgl, klambda, k2, x1, data):

    kxi, kxg, kxgi, n = x
    _, _, I, _, _, Gprod = Glucose_Insulin(kjs, kgj, kjl, kgl, kxi, kxg, kxgi, n, klambda, k2, x1)
    I_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(I_)):
        I_[i] = I[T[i]-1]
    error = np.sum(np.square(I_ - data)) #mean squared error
    
    return error

dataS = np.loadtxt('dataS.csv', delimiter=',')
dataJ = np.loadtxt('dataJ.csv', delimiter=',')
dataL = np.loadtxt('dataL.csv', delimiter=',')
dataGprod = np.loadtxt('dataGprod.csv', delimiter=',')
dataI = np.loadtxt('dataI.csv', delimiter=',')
dataG = np.loadtxt('dataG.csv', delimiter=',')

#stomach
resultS = minimize(objectiveS, x0=0.05, args=(dataS,)) #used to minimize a scalar function of one or more variables
kjs = resultS.x[0]

#Gprod
x0 = [0.1, 18, 15] #initial guesses for klambda, k2, x1
resultGprod = minimize(objectiveGprod, x0, args=(dataGprod, dataG), method='BFGS')
klambda, k2, x1 = resultGprod.x

#Jejenum
x0 = [0.1, 0.1] #initial guesses for kgj and kjl
resultJ = minimize(objectiveJ, x0, args=(kjs, dataJ), method='BFGS')
kgj, kjl = resultJ.x

#Ileum
resultL = minimize(objectiveL, x0=0.05, args=(kjs, kgj, kjl, dataL))
kgl = resultL.x[0]

#Glucose Insulin
x0 = [0.1, 0.1, 0.1, 0.1] #initial guesses for kxi, kxg, kxgi, n
resultGI = minimize(objectiveGI, x0, args=(kjs, kgj, kjl, kgl, klambda, k2, x1, dataI), method='BFGS')
kxi, kxg, kxgi, n = resultGI.x


print("Param | Init | Fitted \n ---------------- \n kjs = ", IP[0], " | ", kjs, " \n kxi = ", IP[1], " | ", kxi, " \n kgj = ", IP[2], " | ", kgj, " \n kjl = ", IP[3], " | ", kjl, " \n kgl = ", IP[4], " | ", kgl, " \n kxg = ", IP[5], " | ", kxg, " \n kxgi = ", IP[6], " | ", kxgi, " \n n = ", IP[7], " | ", n, " \n klambda = ", IP[8], " | ", klambda, " \n k2 = ", IP[9], " | ", k2, " \n x = ", IP[10], " | ", x1, " \n ---------------- \n")
