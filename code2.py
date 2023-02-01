import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
kjs = 0.015
kgj = 0.005
kjl = 0.025
kgl = 0.025
kxg = 0.0005
fgj = 0.5
kxi = 0.1
kxgi = 0.0005
s = 1

Ib = 10
beta = 0.5
Y = 2
Gb = 100

l = 600 #cm, l is the length of the small intestine from the jejunum to the ilium, generally 600-700cm
u = 5 #cm/s, u is the average transit time of the small intestine, generally 0.1-10cm/s
        #velocity will depend on factors such as the contractile force of the intestine, the viscosity of the contents of the intestine, and the diameter of the intestine

tau = l/u

Gprod = 0.1 #mg/s, Gprod is the rate of glucose production in the stomach, generally 0.1-1mg/s
n = 0.1 #mg/s, n is the rate of glucose absorption in the small intestine, generally 0.1-1mg/s

# Initial conditions
S0 = 75
J0 = 0.0
L0 = 0.0
G0 = 0.0
G_ = 0.0
I0 = 0.0

#Gprod = 

# Time points
T = 500
dt = 1.0
time = np.arange(0, T, dt)

t = np.linspace(0, 10, 100)

# Initialize solution arrays
S = np.zeros_like(time)
J = np.zeros_like(time)
L = np.zeros_like(time)
G = np.zeros_like(time)
G_ = np.zeros_like(time)
I = np.zeros_like(time)

S[0] = S0
J[0] = J0
L[0] = L0
G[0] = G0
G_[0] = G0
I[0] = I0


# Solve differential equations
for i in range(1, len(time)):
    dSdt = -kjs*S[i-1]
    dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]

    dGdt = -kxg*G[i-1] - kxgi*G[i-1] + Gprod + n*(kgj*J[i-1] + kgl*L[i-1])

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