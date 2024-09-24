import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate, optimize
import pygad
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit


# Time points
T = 1000
dt = 1.0
time = np.arange(0, T, dt)

# steady-state values
Ib = 10
Gb = 100 

l = 300 #cm, l is the length of the small intestine from the jejunum to the ilium, generally 600-700cm
u = 5 #cm/s, u is the average transit time of the small intestine, generally 0.1-10cm/s
        #velocity will depend on factors such as the contractile force of the intestine, the viscosity of the contents of the intestine, and the diameter of the intestine

tau = l/u

# Initial conditions
S0 = 75
J0 = 0.0
L0 = 0.0

G0 = 25
I0 = 12

Gprod0 = 0

dataS = np.loadtxt('Data/dataS.csv', delimiter=',')
dataJ = np.loadtxt('Data/dataJ.csv', delimiter=',')
dataL = np.loadtxt('Data/dataL.csv', delimiter=',')
dataGprod = np.loadtxt('Data/dataGprod.csv', delimiter=',')
dataI = np.loadtxt('Data/dataI.csv', delimiter=',')
dataG = np.loadtxt('Data/dataG.csv', delimiter=',')

#----------
#Stomach
#----------

def Stomach(kjs):
    S = np.zeros_like(time)
    S[0] = S0
    for i in range(1, len(time)):
        dSdt = -kjs*S[i-1]
        S[i] = S[i-1] + dSdt*dt
    return S

def objectiveS(x, data):
    S = Stomach(x)
    S_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(S_)):
        S_[i] = S[T[i]-1]
    error = np.sum(np.square(S_ - data)) #mean squared error
    
    return error

resultS = minimize(objectiveS, x0=0.05, args=(dataS,)) #used to minimize a scalar function of one or more variables
kjs = resultS.x[0]

#----------
#Insulin
#----------

def Insulin(kxi):
    I = np.zeros_like(time)
    I[0] = I0
    for i in range(1, len(time)):
        if i < 30:
            dIdt = 0
        else:
            dIdt = kxi * (- I[i-1])
        I[i] = I[i-1] + dIdt*dt

    return I

def objectiveI(x, data):
    I = Insulin(x)
    I_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(I_)):
        I_[i] = I[T[i]-1]
    error = np.sum(np.square(I_ - data)) #mean squared error
    
    return error

resultI = minimize(objectiveI, x0=0.05, args=(dataI,)) #used to minimize a scalar function of one or more variables
kxi = resultI.x[0]

#----------
#Jejunum
#----------

def Jejunum(kjs, kgj, kjl):
    J = np.zeros_like(time)
    J[0] = J0
    S = Stomach(kjs)
    for i in range(1, len(time)):
        dJdt = kjs*S[i-1] - kgj*J[i-1] - kjl*J[i-1]
        J[i] = J[i-1] + dJdt*dt
    return J

def objectiveJ(params, data):
    kgj, kjl = params
    J = Jejunum(kjs, kgj, kjl)
    J_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(J_)):
        J_[i] = J[T[i]-1]
    error = np.sum(np.square(J_ - data))  # Mean squared error
    return error

def jejunum_optimization(initial_guesses, data):
    best_result = None
    best_params = None
    best_error = float('inf')

    for initial_guess in initial_guesses:
        result = minimize(objectiveJ, x0=initial_guess, args=(data,), method='L-BFGS-B', bounds=[(0.01, 0.1), (0.001, 0.01)])
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
            best_result = result

    return best_result, best_params

initial_guesses = [[0.05, 0.01], [0.07, 0.005], [0.03, 0.008]]  # Add more diverse initial guesses
best_result, best_params = jejunum_optimization(initial_guesses, dataJ)

kgj, kjl = best_params
print("Best estimated parameters: kgj =", kgj, "kjl =", kjl)


#----------
#Ileum
#----------

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


def objectiveL(x, data):
    L = Ileum(kjs, kgj, kjl, x)
    L_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(L_)):
        L_[i] = L[T[i]-1]
    error = np.sum(np.square(L_ - data)) #mean squared error
    
    return error

def ileum_optimization(initial_guesses, data):
    best_result = None
    best_params = None
    best_error = float('inf')

    for initial_guess in initial_guesses:
        result = minimize(objectiveL, x0=initial_guess, args=(data,), method='L-BFGS-B', bounds=[(0.001, 0.1)])
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
            best_result = result

    return best_result, best_params

initial_guesses = [[0.02], [0.015], [0.025], [0.01], [0.03]] 
best_result, best_params = ileum_optimization(initial_guesses, dataL)

kgl = best_params[0]


#----------
#GI
#----------

def Glucose_Insulin(kgj, kjl, kgl, kxg, kxgi, n, klambda, k2, x):
    G = np.zeros_like(time)
    G[0] = G0
    Gprod = np.zeros_like(time)
    Gprod[0] = Gprod0
    J = Jejunum(kjs, kgj, kjl)
    L = Ileum(kjs, kgj, kjl, kgl)
    S = Stomach(kjs)
    I = Insulin(kxi)

    for i in range(1, len(time)):
        Gprod[i] = (klambda * (Gb-G[i-1])) / (k2 + (Gb - x)) + Gprod[0] 
        if i < 30:
            dGdt = -kxg*G[i-1] + Gprod[i-1] + n*(kgj*J[i-1] + kgl*L[i-1])
        else:
            dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod[i-1] + n*(kgj*J[i-1] + kgl*L[i-1])
        G[i] = G[i-1] + dGdt*dt # when dgdt*dt becomes <0 find value of dgdt. this gives natural loss of glucose in the blood.

    return S, G, I, J, L, Gprod

def objective(params, data):
    kxg, kxgi, n, klambda, k2, x = params
    _, G, _, _, _, _ = Glucose_Insulin(kgj, kjl, kgl, kxg, kxgi, n, klambda, k2, x)
    error = np.sum(np.square(G[:len(data)] - data))  # Mean squared error
    return error


#kxg, kxgi, n, klambda, k2, x
# Initial guesses for the parameters
x0 = [0.01, 0.01, 0.01, 0.1, 17, 17]

# Bounds for the parameters
bounds = [(0.001, 0.05), (0.001, 0.05), (0.001, 0.05), (0.1, 1), (5, 30), (5, 30)]

# Minimize the objective function
result = minimize(objective, x0=x0, args=(dataG,), bounds=bounds, method='L-BFGS-B')

# Extract optimized parameters
kxg, kxgi, n, klambda, k2, x = result.x

S_Optimized = Stomach(kjs)
I_Optimized = Insulin(kxi)
J_Optimized = Jejunum(kjs, kgj, kjl)
L_Optimized = Ileum(kjs, kgj, kjl, kgl)
_, GI_Optimized, _, _, _, _ = Glucose_Insulin(kgj, kjl, kgl, kxg, kxgi, n, klambda, k2, x)


IP = [0.034, 0.025, 0.067, 0.007, 0.02, 0.018, 0.028, 0.01, 0.6, 22, 16]
print("Param | Init | Fitted \n ---------------- \n kjs | ", IP[0], " | ", kjs, " \n kxi | ", IP[1], " | ", kxi, " \n kgj | ", IP[2], " | ", kgj, " \n kjl | ", IP[3], " | ", kjl, " \n kgl | ", IP[4], " | ", kgl, " \n kxg | ", IP[5], " | ", kxg, " \n kxgi | ", IP[6], " | ", kxgi, " \n n | ", IP[7], " | ", n, " \n klambda | ", IP[8], " | ", klambda, " \n k2 | ", IP[9], " | ", k2, " \n x | ", IP[10], " | ", x, " \n ---------------- \n")

# Data and labels
data = [(S_Optimized, dataS, 'Stomach'),
        (I_Optimized, dataI, 'Insulin'),
        (J_Optimized, dataJ, 'Jejunum'),
        (L_Optimized, dataL, 'Ileum'),
        (GI_Optimized, dataG, 'Glucose-Insulin')]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot each data set in a subplot
for ax, (optimized, actual, label) in zip(axs.flat, data):
    ax.plot(time, optimized, label=f'Fitted {label}', color='blue')
    ax.scatter(time[:len(actual)], actual, label='Actual Data', color='red', marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.set_title(f'{label} with Fitted Parameters vs Actual Data')
    ax.legend()
    ax.grid(True)

# Remove the last empty subplot
fig.delaxes(axs[1, 2])

# Adjust layout
plt.tight_layout()
plt.show()