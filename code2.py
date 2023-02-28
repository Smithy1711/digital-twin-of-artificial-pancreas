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

#Gprod = 0.1 #mg/s, Gprod is the rate of glucose production in the stomach, generally 0.1-1mg/s
n = 0.6 #mg/s, n is the rate of glucose absorption in the small intestine, generally 0.1-1mg/s

# Initial conditions
S0 = 100
J0 = 0.0
L0 = 0.0
I0 = 0.0


# Time points
T = 1000
dt = 1.0
time = np.arange(0, T, dt)

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

#all equations
'''
dSdt = -kjs*S[i-1]
dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]
dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod + n*(kgj*J[i-1] + kgl*L[i-1])
dLdt = kjl * phi - kgl * L[i-1]
dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib)
'''


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

def Glucose_Insulin(kjs):
    G0 = 25
    G_ = np.zeros_like(time)
    G_[0] = G0
    I = np.zeros_like(time)
    I[0] = I0
    G = np.zeros_like(time)
    G[0] = G0
    J = Jejunum(kjs)
    L = Ileum(kjs)

    for i in range(1, len(time)):
        Gprod = 1/(1+G[i-1])
        G_[i] = G[i-1] + np.sum(fgj * (kgj * J[i-1] + kgl * L[i-1]))
        dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod + n*(kgj*J[i-1] + kgl*L[i-1])
        G[i] = G[i-1] + dGdt*dt
        #dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib) #look at each part of the equation in the paper
        # beta and gamma are not needed in a diabetic person because they are not producing insulin. beta and gamma are parameters for half saturation and acceleration of the insulin production 
        # insulin can take up to 30 mins before it starts to work. 1 to 3 hours for the peak activity. https://www.diabetes.co.uk/insulin/insulin-actions-and-durations.html
        # can replace the acceleration with a delay? also may need to account for basal insulin or "left over insulin" in the body
        # dIdt = kxi * Ib * ((1) / (((Gb/G_[i-1])) + 1) - I[i-1]/Ib)
        if i < 30:
            dIdt = 0
        else:
            dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib)
        I[i] = I[i-1] + dIdt*dt

    return G, I, J, L

def plot(G, I, J , L):
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
    kjs = 0.01172569
    G, I, J, L = Glucose_Insulin(kjs)
    return G, I, J, L

G, I, J, L = main()
print("Insulin Dose: ", I[120], "uU/mL")
plot(G, I, J, L)

def cost_function(alpha, Gexp, Iexp, Gmax, Gmin, Imax, Imin):
        G, I = Glucose_Insulin()
        T = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120]
        J = 1 / (5 * (Gmax - Gmin) ** 2) * np.sum((Gexp - G[T[1]]) ** 2)
        J += alpha / (5 * (Imax - Imin) ** 2) * np.sum((Iexp - I[T[1]]) ** 2)
        return J


#print(cost_function(1, 150, 7.1, 39, 373, 8.6, 5.9))



def objective(kjs, data):
    S = Stomach(kjs)
    S_ = np.zeros(len(data))
    T = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120]
    for i in range(0, len(S_)):
        S_[i] = S[T[i]-1]
    error = np.sum(np.square(S_ - data)) #mean squared error
    
    return error

data = np.loadtxt('data.txt')
print(data)

#result is 3.47% error
result = minimize(objective, x0=0.087, args=(data,)) #used to minimize a scalar function of one or more variables
print(result.x)
sRes = Stomach(result.x)


S_ = np.zeros(len(data))
T = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120]
for i in range(0, len(S_)):
    S_[i] = sRes[T[i]-1]
print(S_)



bounds = [(0.0, 0.1)] # specify the bounds for kjs
result2 = differential_evolution(objective, bounds=bounds, args=(data,)) #multi-dimensional, continuous function. used to find the global minima of a function
print(result2.x)


''''
class GlucoseInsulin:
    def __init__(self, kjs):
        self.kjs = kjs
        self.S0 = 100
        self.J0 = 0.0
        self.L0 = 0.0
        self.I0 = 0.0
        self.G0 = 25


        self.S = np.zeros_like(time)
        self.S[0] = self.S0

        self.J = np.zeros_like(time)
        self.J[0] = self.J0

        self.G = np.zeros_like(time)
        self.G[0] = self.G0

        self.I = np.zeros_like(time)
        self.I[0] = self.I0

        self.L = np.zeros_like(time)
        self.L[0] = self.L0


    def stomach(self):
        
        for i in range(1, len(time)):
            dSdt = -self.kjs * self.S[i-1]
            self.S[i] = self.S[i-1] + dSdt * dt

    def jejunum(self):
        gi = GlucoseInsulin(0.01172569)
        S = gi.stomach()
        print(S)
        print(self.stomach())
        for i in range(1, len(time)):
            dJdt = self.kjs * S[i-1] - kgj * self.J[i-1] - kjl * self.J[i-1]
            self.J[i] = self.J[i-1] + dJdt * dt

    def ileum(self):
        gi = GlucoseInsulin(0.01172569)
        J = gi.jejunum()
        for i in range(1, len(time)):
            if time[i] < tau:
                phi = 0
            else:
                phi = J[np.floor(i-tau).astype(int)]
            dLdt = kjl * phi - kgl * self.L[i-1]
            self.L[i] = self.L[i-1] + dLdt * dt

    def glucose_insulin(self):
        gi = GlucoseInsulin(0.01172569)
        L = gi.ileum()
        J = gi.jejunum()
        for i in range(1, len(time)):
            Gprod = 0.1  # mg/s
            G_ = self.G[i-1] + np.sum(fgj * (kgj * J + kgl * L))
            dGdt = -kxg * self.G[i-1] - kxgi * self.G[i-1] * self.I[i-1] + Gprod + n * (kgj * J[i-1] + kgl * L[i-1])
            dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_))**Y + 1) - self.I[i-1]/Ib)
            # beta and gamma are not needed in a diabetic person because they are not producing insulin. beta and gamma are parameters for half saturation and acceleration of the insulin production 
            # dIdt = kxi * Ib * ((1) / (((Gb/G_[i-1])) + 1) - I[i-1]/Ib)
            self.G[i] = self.G[i-1] + dGdt * dt
            self.I[i] = self.I[i-1] + dIdt * dt

        return self.G, self.I


gi = GlucoseInsulin(0.01172569)

L = gi.ileum()
J = gi.jejunum()
G, I = gi.glucose_insulin()


#Amount of glucose in the jejunum over time
plt.subplot(2, 2, 1)
plt.plot(time, J, label='J(t)')
plt.xlabel('Time (s)')
plt.ylabel('J(t)')

#Amount of glucose in the ilium over time
plt.subplot(2, 2, 2)
plt.plot(time, L)
plt.xlabel('Time (s)')
plt.ylabel('L(t)')

#Glucose concentration in the blood over time
plt.subplot(2, 2, 3)
plt.plot(time, G)
plt.xlabel('Time (s)')
plt.ylabel('G(t)')

#Insulin concentration in the blood over time
plt.subplot(2, 2, 4)
plt.plot(time, I)
plt.xlabel('Time (s)')
plt.ylabel('I(t)')

plt.show()
'''