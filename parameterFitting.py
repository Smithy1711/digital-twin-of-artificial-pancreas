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


def Glucose_Insulin(kjs, kgj, kjl, kgl, kxi, kxg, kxgi, n, klambda, k2, x):
    #G_ = np.zeros_like(time)
    #G_[0] = G0
    G = np.zeros_like(time)
    G[0] = G0
    I = Insulin(kxi)
    S = Stomach(kjs)
    J = Jejunum(kjs, kgj, kjl)
    L = Ileum(kjs, kgj, kjl, kgl)
    Gprod = np.zeros_like(time)


    for i in range(1, len(time)):
        Gprod[i] = (klambda * (Gb-G[i-1])) / (k2 + (Gb - x)) + Gprod[0] 
        
        '''#klambda - mM2 min−1 Kinetic constant for hepatic glucose release rate

        
        #G_[i] = G[i-1] + np.sum(fgj * (kgj * J[i-1] + kgl * L[i-1])) not needed in a person with type 1 diabetes

        
        # when a value >0 is given for G0 then dGdt will give the natural loss of glucose.
        # also making there be no bolus then J and L are 0. also I is 0 as there is no administration of insuline
        # therefore dGdt is just -kxg*G[i-1]

        # dIdt = kxi * Ib * ((beta**Y + 1) / ((beta**Y * (Gb/G_[i-1]))**Y + 1) - I[i-1]/Ib) #look at each part of the equation in the paper
        # beta and gamma are not needed in a diabetic person because they are not producing insulin. beta and gamma are parameters for half saturation and acceleration of the insulin production 
        # insulin can take up to 30 mins before it starts to work. 1 to 3 hours for the peak activity. https://www.diabetes.co.uk/insulin/insulin-actions-and-durations.html
        # can replace the acceleration with a delay? also may need to account for basal insulin or "left over insulin" in the body
        # dIdt = kxi * Ib * ((1) / (((Gb/G_[i-1])) + 1) - I[i-1]/Ib)'''
        if i < 30:
            dGdt = -kxg*G[i-1] + Gprod[i-1] + n*(kgj*J[i-1] + kgl*L[i-1])
        else:
            dGdt = -kxg*G[i-1] - kxgi*G[i-1]*I[i-1] + Gprod[i-1] + n*(kgj*J[i-1] + kgl*L[i-1])
        G[i] = G[i-1] + dGdt*dt # when dgdt*dt becomes <0 find value of dgdt. this gives natural loss of glucose in the blood.

    return S, G, I, J, L, Gprod

dataS = np.loadtxt('Data/dataS.csv', delimiter=',')
dataJ = np.loadtxt('Data/dataJ.csv', delimiter=',')
dataL = np.loadtxt('Data/dataL.csv', delimiter=',')
dataGprod = np.loadtxt('Data/dataGprod.csv', delimiter=',')
dataI = np.loadtxt('Data/dataI.csv', delimiter=',')
dataG = np.loadtxt('Data/dataG.csv', delimiter=',')



#parameter fitting

def objectiveS(x, data):
    S = Stomach(x)
    S_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(S_)):
        S_[i] = S[T[i]-1]
    error = np.sum(np.square(S_ - data)) #mean squared error
    
    return error

#stomach
resultS = minimize(objectiveS, x0=0.05, args=(dataS,)) #used to minimize a scalar function of one or more variables
kjs = resultS.x[0]

#insulin supplimentary function for parameter fitting
def insulin(kxi):
    I = np.zeros_like(time)
    I[0] = I0
    for i in range(1, len(time)):
        if time[i] < 30:
            dIdt = 0
        else:
            dIdt = kxi * (- I[i-1])
        I[i] = I[i-1] + dIdt*dt
    return I

#insulin parameter fitting
def objectiveI(x, data):
    I = insulin(x)
    I_ = np.zeros(len(data))
    T = list(range(1, len(data)+1))
    for i in range(0, len(I_)):
        I_[i] = I[T[i]-1]
    error = np.sum(np.square(I_ - data)) #mean squared error
    
    return error

resultI = minimize(objectiveI, x0=0.05, args=(dataI,)) #used to minimize a scalar function of one or more variables
kxi = resultI.x[0]

def error_function(params):
    # Get the model predictions using the current parameters
    S, G, I, J, L, Gprod = Glucose_Insulin_Fixed(*params)

    # Calculate the error (sum of squared differences) between the model predictions and the data
    error_S = np.sum((S - dataS)**2)
    error_G = np.sum((G - dataG)**2)
    error_I = np.sum((I - dataI)**2)
    error_J = np.sum((J - dataJ)**2)
    error_L = np.sum((L - dataL)**2)
    error_Gprod = np.sum((Gprod - dataGprod)**2)

    # Return the total error
    return error_S + error_G + error_I + error_J + error_L + error_Gprod
    #return error_G

def Glucose_Insulin_Fixed(kgj, kjl, kgl, kxg, kxgi, n, klambda, k2, x):
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


# Set the bounds for the parameters
bounds = [(0.01, 0.1), (0.004, 0.01), (0.01, 0.06), (0.01, 0.06), (0.01, 0.05), (0.005, 0.05), (0.4, 0.8), (12, 25), (12, 26)]

# Use differential_evolution to find the best parameters
result = differential_evolution(error_function, bounds, strategy='best1bin', popsize=50, mutation=(0.5, 1), recombination=0.7, tol=0.01)
print(result)

# Get the optimal parameters from the result
optimal_params = result.x


kgj = optimal_params[0]
kjl = optimal_params[1]
kgl = optimal_params[2]
kxg = optimal_params[3]
kxgi = optimal_params[4]
n = optimal_params[5]
klambda = optimal_params[6]
k2 = optimal_params[7]
x = optimal_params[8]

IP = [0.034, 0.025, 0.067, 0.007, 0.02, 0.018, 0.028, 0.01, 0.6, 22, 16]
print("Param | Init | Fitted \n ---------------- \n kjs | ", IP[0], " | ", kjs, " \n kxi | ", IP[1], " | ", kxi, " \n kgj | ", IP[2], " | ", kgj, " \n kjl | ", IP[3], " | ", kjl, " \n kgl | ", IP[4], " | ", kgl, " \n kxg | ", IP[5], " | ", kxg, " \n kxgi | ", IP[6], " | ", kxgi, " \n n | ", IP[7], " | ", n, " \n klambda | ", IP[8], " | ", klambda, " \n k2 | ", IP[9], " | ", k2, " \n x | ", IP[10], " | ", x, " \n ---------------- \n")



# Run the simulation with the optimal parameters and compare the results with the data
S_opt, G_opt, I_opt, J_opt, L_opt, Gprod_opt = Glucose_Insulin(kjs, kgj, kjl, kgl, kxi, kxg, kxgi, n, klambda, k2, x)



def plot_simulation_vs_experiment(time, sim_data, exp_data, labels, ylabels, xlabel=None):
    plt.figure(figsize=(12, 12))
    for i, (sim, exp, label, ylabel) in enumerate(zip(sim_data, exp_data, labels, ylabels), 1):
        plt.subplot(3, 2, i)
        plt.plot(time, sim, label=f"Simulated {label}")
        plt.plot(time, exp, label=f"Experimental {label}")
        plt.ylabel(ylabel)
        plt.legend()
        if i > 4:
            plt.xlabel(xlabel)
    plt.show()

simulated_data = [G_opt, I_opt, J_opt, L_opt, Gprod_opt, S_opt]
experimental_data = [dataG, dataI, dataJ, dataL, dataGprod, dataS]
labels = ["Glucose", "Insulin", "Jejunum", "Ileum", "Gprod", "Stomach"]
ylabels = ["Glucose Concentration", "Insulin Concentration", "Jejunum Concentration", "Ileum Concentration", "Gprod Concentration", "Stomach Concentration"]

plot_simulation_vs_experiment(time, simulated_data, experimental_data, labels, ylabels, xlabel="Time")
