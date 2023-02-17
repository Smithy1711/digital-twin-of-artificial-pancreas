import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pygad
from scipy.optimize import minimize


# Constants

#to be determined by optimization
kgj = 0.005
kjl = 0.025
kgl = 0.025
kxg = 0.0005
fgj = 0.5
kxi = 0.1
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

Gprod = 0.1 #mg/s, Gprod is the rate of glucose production in the stomach, generally 0.1-1mg/s
n = 0.6 #mg/s, n is the rate of glucose absorption in the small intestine, generally 0.1-1mg/s

# Initial conditions
S0 = 100
J0 = 0.0
L0 = 0.0
G0 = 196
I0 = 0.0


# Time points
T = 120
dt = 1.0
time = np.arange(0, T, dt)

t = np.linspace(0, 10, 100)

# Solve differential equations
'''
def model():
    for i in range(1, len(time)):
        dSdt = -kjs*S[i-1]
        dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]

        dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod + n*(kgj*J[i-1] + kgl*L[i-1])

        # Calculate phi at current time step
        if time[i] < tau:
            phi = 0
        else:
            phi = J[np.floor(i-tau).astype(int)]
        # Calculate derivative at current time step
        dLdt = kjl * phi - kgl * L[i-1]
        
        G_[i] = G[i-1] + np.sum(fgj * (kgj * J + kgl * L))
        dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib)
        

        # Update solution at current time step
        S[i] = S[i-1] + dSdt*dt
        J[i] = J[i-1] + dJdt*dt
        L[i] = L[i-1] + dLdt*dt
        G[i] = G[i-1] + dGdt*dt
        I[i] = I[i-1] + dIdt*dt

    return G, I
'''
def Stomach(kjs):
    S = np.zeros_like(time)
    S[0] = S0
    for i in range(1, len(time)):
        dSdt = -kjs*S[i-1]
        S[i] = S[i-1] + dSdt*dt
    return S

def Jejunum():
    J = np.zeros_like(time)
    J[0] = J0
    S = Stomach()
    for i in range(1, len(time)):
        dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]
        J[i] = J[i-1] + dJdt*dt
    return J

def Ileum():
    L = np.zeros_like(time)
    L[0] = L0
    J = Jejunum()
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

def Glucose_Insulin():
    G0 = 196
    G_ = np.zeros_like(time)
    G_[0] = G0
    I = np.zeros_like(time)
    I[0] = I0
    G = np.zeros_like(time)
    G[0] = G0
    J = Jejunum()
    L = Ileum()

    for i in range(1, len(time)):
        G_[i] = G[i-1] + np.sum(fgj * (kgj * J + kgl * L))
        dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib)
        I[i] = I[i-1] + dIdt*dt
        dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod + n*(kgj*J[i-1] + kgl*L[i-1])
        G[i] = G[i-1] + dGdt*dt

    return G, I
    

def cost_function(alpha, Gexp, Iexp, Gmax, Gmin, Imax, Imin):
        G, I = Glucose_Insulin()
        T = [0, 30, 60, 90, 120]
        J = 1 / (5 * (Gmax - Gmin) ** 2) * np.sum((Gexp - G[T[1]]) ** 2)
        J += alpha / (5 * (Imax - Imin) ** 2) * np.sum((Iexp - I[T[1]]) ** 2)
        return J


#print(cost_function(1, 150, 7.1, 39, 373, 8.6, 5.9))



def objective(kjs, data):
    S = Stomach(kjs)
    S_ = np.zeros(len(data))
    T = [1, 30, 60, 90, 120]
    for i in range(0, len(S_)):
        S_[i] = S[T[i]-1]
    error = np.sum(np.square(S_ - data))
    return error

data = np.loadtxt('data.txt')

result = minimize(objective, x0=0.087, args=(data,))
print(result.x)
print(objective(result.x, data))

from scipy.optimize import differential_evolution

bounds = [(0.0, 0.1)] # specify the bounds for kjs
result2 = differential_evolution(objective, bounds=bounds, args=(data,))
print(result2.x)

'''
    # Plot solutions

    #Bolus value in the stomach over time
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 3, 1)
    plt.plot(time, S, label='S(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('S(t)')

    #Amount of glucose in the jejunum over time
    plt.subplot(2, 3, 2)
    plt.plot(time, J, label='J(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('J(t)')

    #Amount of glucose in the ilium over time
    plt.subplot(2, 3, 3)
    plt.plot(time, L)
    plt.xlabel('Time (s)')
    plt.ylabel('L(t)')

    #Glucose concentration in the blood over time
    plt.subplot(2, 3, 4)
    plt.plot(time, G)
    plt.xlabel('Time (s)')
    plt.ylabel('G(t)')

    #Insulin concentration in the blood over time
    plt.subplot(2, 2, 4)
    plt.plot(time, I)
    plt.xlabel('Time (s)')
    plt.ylabel('I(t)')


    plt.show()

    values = np.hstack((S[:, np.newaxis], J[:, np.newaxis], L[:, np.newaxis], G[:, np.newaxis], I[:, np.newaxis]))

    # Save data to CSV files
    np.savetxt("values.csv", values, delimiter=",")

class Model:

    # Initial conditions

    T = 500
    dt = 1.0
    time = np.arange(0, T, dt)

    t = np.linspace(0, 10, 100)
    
    kjs = 0.015
    kgj = 0.005
    kjl = 0.025
    kgl = 0.025
    kxg = 0.0005
    fgj = 0.5
    kxi = 0.1
    kxgi = 0.0005



    def __int__(self, S0, J0, L0, G0, G_0, I0, l, u, tau, Gprod, n, beta, Y, Ib, Gb):
        self.S0 = S0
        self.J0 = J0
        self.L0 = L0
        self.G0 = G0
        self.G_0 = G_0
        I0 = I0
        self.l = l
        self.u = u
        self.tau = tau
        self.Gprod = Gprod
        self.n = n
        self.beta = beta
        self.Y = Y
        self.Ib = Ib
        self.Gb = Gb

    def 

    def Stomach(self):
        for i in range(1, len(self.time)):
            dSdt = -self.kjs*S[i-1]
            self.S[i] = self.S[i-1] + dSdt*dt
        return S

    def Jejunum(self):
        for i in range(1, len(time)):
            dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]
            J[i] = J[i-1] + dJdt*dt
        return J

    def Ileum(self):
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

    def Glucose(self):
        for i in range(1, len(time)):
            dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod + n*(kgj*J[i-1] + kgl*L[i-1])
            G[i] = G[i-1] + dGdt*dt
        return G

    def Insulin(self):
        G0 = 196
        G_ = np.zeros_like(self.time)
        G_[0] = G0
        for i in range(1, len(time)):
            G_[i] = G[i-1] + np.sum(fgj * (kgj * J + kgl * L))
            dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib)
            I[i] = I[i-1] + dIdt*dt
        return I

    def cost_function(alpha, Gexp, Iexp, Gmax, Gmin, Imax, Imin):
        G = Glucose()
        I = Insulin()
        T = [0, 30, 60, 90, 120]
        J = 1 / (5 * (Gmax - Gmin) ** 2) * np.sum((Gexp - G[T[1]]) ** 2)
        J += alpha / (5 * (Imax - Imin) ** 2) * np.sum((Iexp - I[T[1]]) ** 2)
        return J


    print(cost_function(1, 150, 7.1, 39, 373, 8.6, 5.9))

'''