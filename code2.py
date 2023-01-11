import numpy as np
import matplotlib.pyplot as plt

# Constants
kjs = 0.015
kgj = 0.005
kjl = 0.025
kgl = 0.025
kxg = 0.0005
kxgi = 0.0005
s = 1

l = 600 #cm, l is the length of the small intestine from the jejunum to the ilium, generally 600-700cm
u = 5 #cm/s, u is the average transit time of the small intestine, generally 0.1-10cm/s
        #velocity will depend on factors such as the contractile force of the intestine, the viscosity of the contents of the intestine, and the diameter of the intestine

tau = l/u


# Initial conditions
S0 = 75
J0 = 0.0
L0 = 0.0
Gb = 0.0

#Gprod = 

# Time points
T = 500
dt = 1.0
time = np.arange(0, T, dt)

# Initialize solution arrays
S = np.zeros_like(time)
J = np.zeros_like(time)
L = np.zeros_like(time)
G = np.zeros_like(time)

S[0] = S0
J[0] = J0
L[0] = L0
G[0] = Gb

# Solve differential equations
for i in range(1, len(time)):
    dSdt = -kjs*S[i-1]
    dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]

    dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I + Gprod + n*(kgj*J + kgl*L) #issue***
    G[i] = G[i-1] + dGdt*dt

    # Calculate phi at current time step
    if time[i] < tau:
        phi = 0
    else:
        phi = J[int(i-tau)]
    # Calculate derivative at current time step
    dLdt = kjl * phi - kgl * L[i-1]
    
    # Update solution at current time step
    S[i] = S[i-1] + dSdt*dt
    J[i] = J[i-1] + dJdt*dt
    L[i] = L[i-1] + dLdt*dt

# Plot solutions

#Bolus value in the stomach over time
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.plot(time, S, label='S(t)')
plt.xlabel('Time (s)')
plt.ylabel('S(t)')

#Amount of glucose in the jejunum over time
plt.subplot(2, 2, 2)
plt.plot(time, J, label='J(t)')
plt.xlabel('Time (s)')
plt.ylabel('J(t)')
plt.legend()

#Amount of glucose in the ilium over time
plt.subplot(2, 2, 3)
plt.plot(time, L)
plt.xlabel('Time (s)')
plt.ylabel('L(t)')


plt.show()

np.savetxt("jejunum.csv", J, delimiter=",")
np.savetxt("ileum.csv", L, delimiter=",")
np.savetxt("stomach.csv", S, delimiter=",")