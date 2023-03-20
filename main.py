import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pygad
from scipy.optimize import minimize
from scipy.optimize import differential_evolution





# Constants

#to be determined by optimization
kgj = 0.005
kjl = 0.025
kgl = 0.025
kxg = 0.0005
fgj = 0.005
kxgi = 0.0005

beta = 0.5
Y = 0.04

# steady-state values
Ib = 10
Gb = 100 

l = 600 #cm, l is the length of the small intestine from the jejunum to the ilium, generally 600-700cm
u = 5 #cm/s, u is the average transit time of the small intestine, generally 0.1-10cm/s
        #velocity will depend on factors such as the contractile force of the intestine, the viscosity of the contents of the intestine, and the diameter of the intestine

tau = l/u

#Gprod = 0.1 #mg/s, Gprod is the rate of glucose production in the stomach, generally 0.1-1mg/s
n = 0.6 #mg/s, n is the rate of glucose absorption in the small intestine, generally 0.1-1mg/s

# Initial conditions
S0 = 75
J0 = 0.0
L0 = 0.0

G0 = 25
I0 = 12

Gprod0 = 2



# Time points
T = 1000
dt = 1.0
time = np.arange(0, T, dt)

GI_ratio = 8


def Stomach(kjs):
    S = np.zeros_like(time)
    S[0] = S0
    for i in range(1, len(time)):
        dSdt = -kjs*S[i-1]
        S[i] = S[i-1] + dSdt*dt
    return S

def Jejunum(kjs):
    J = np.zeros_like(time)
    J[0] = J0
    S = Stomach(kjs)
    for i in range(1, len(time)):
        dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]
        J[i] = J[i-1] + dJdt*dt
    return J

def Ileum(kjs):
    L = np.zeros_like(time)
    L[0] = L0
    J = Jejunum(kjs)
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

def Glucose_Insulin(kjs, kxi, klambda, k2, x, G0, I0, Gprod0):
    #G_ = np.zeros_like(time)
    #G_[0] = G0
    I = np.zeros_like(time)
    I[0] = I0
    G = np.zeros_like(time)
    G[0] = G0
    Gprod = np.zeros_like(time)
    Gprod[0] = Gprod0
    J = Jejunum(kjs)
    L = Ileum(kjs)
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

def plot(G, I,J ,L):
    #Amount of glucose in the jejunum over time
    plt.subplot(2, 2, 1)
    plt.plot(time, J, label='J(t)')
    plt.xlabel('Time (min)')
    plt.ylabel('J(t)')

    #Amount of glucose in the ilium over time
    plt.subplot(2, 2, 2)
    plt.plot(time, L)
    plt.xlabel('Time (min)')
    plt.ylabel('L(t)')

    #Glucose concentration in the blood over time
    plt.subplot(2, 2, 3)
    plt.plot(time, G)
    plt.xlabel('Time (min)')
    plt.ylabel('G(t)')

    #Insulin concentration in the blood over time
    plt.subplot(2, 2, 4)
    plt.plot(time, I)
    plt.xlabel('Time (min)')
    plt.ylabel('I(t)')

    plt.show()

def main():
    kjs = 0.034
    kxi = 0.025
    klambda = 0.02
    k2 = 22
    x = 16
    S, G, I, J, L, Gprod = Glucose_Insulin(kjs, kxi, klambda, k2, x, G0, I0, Gprod0)
    return S, G, I, J, L, Gprod

S, G, I, J, L, Gprod = main()

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

def objectiveI(x, data, kjs, klambda, k2, x1):
    S, G, I, J, L, Gprod = Glucose_Insulin(kjs, x, klambda, k2, x1, G0, I0, Gprod0)
    I_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(I_)):
        I_[i] = I[T[i]-1]
    error = np.sum(np.square(I_ - data)) #mean squared error
    
    return error

def objectiveGprod(klambda, k2, x1, data, G):
    #klambda, k2, x1 = x
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


dataS = np.loadtxt('dataS.csv')
resultS = minimize(objectiveS, x0=0.05, args=(dataS,)) #used to minimize a scalar function of one or more variables

dataGprod = np.loadtxt('dataGprod.csv')
G = np.loadtxt('dataG.csv')
#resultGprod = minimize(objectiveGprod, x0=[0.1,5,5], args=(dataGprod, G), method='Nedler-Mead') #used to minimize a scalar function of one or more variables
# Define bounds for the parameters (optional)
bounds = [(0, None), (0, None), (0, None)]

fun = lambda x: objectiveGprod(x[0], x[1], x[2], dataGprod, G)
x0 = [0.1,5,5]
resultGprod = minimize(fun, x0, bounds=bounds)


klambda, k2, x1 = resultGprod.x
dataI = np.loadtxt('dataI.csv')
resultI = minimize(objectiveI, x0=0.01, args=(dataI,resultS.x,klambda,k2,x1)) #used to minimize a scalar function of one or more variables

kjs = resultS.x[0]
kxi = resultI.x[0]


print('kjs =', kjs)
print('kxi =', kxi)
print('klambda =', klambda)
print('k2 =', k2)
print('x1 =', x1)


