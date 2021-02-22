import multiprocessing as mp
import numpy as np
import pandas as pd
import math
import csv
import os
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from EpiModels import SECIR_vacc_model_hospital_age
from reset_parameters import initializeForVaccineStates

def runScenarios(inputList):
    state, eV = inputList
    model = SECIR_vacc_model_hospital_age
    duration = 365
    t = np.linspace(0, duration, duration + 1)
    
    # load fitted parameters
    filename = state+'_since_may_fixed_params.csv'
    opt_params = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            opt_params.append([float(el) for el in row])
    
    theta, alphaC = opt_params[0][13:15]
    theta = np.full(4, theta)
    
    output_may21 = []
    for lambdaV in np.arange(0.1, 1.01, 0.1):
        for sc in range(4):
            if sc == 0:
                vacc_type = 'I0-S0'
                alphaP = 0
                thetaV = np.full(4, 0)
            elif sc == 1:
                vacc_type = 'I0-S1'
                alphaP = 0
                thetaV = theta
            elif sc == 2:
                vacc_type = 'I1-S0'
                alphaP = alphaC
                thetaV = np.full(4, 0)          
            elif sc == 3:
                vacc_type = 'I1-S1'
                alphaP = alphaC
                thetaV = theta

            for t0V in [210, 300]:
                for omegaV in [1/90, 1/180, 1/270, 1/360]:
                    for phiV in [1/60, 1/120]:
                        N, S0, E0, C0, IN0, IH0, RI0, RC0, CumC0, CumIN0, CumIH0, D0 = opt_params[0][1:13]
                        theta, alphaC, beta, gammaC, gammaIN, gammaIH, omegaC, omegaI, mu, sigma, rho, hospRate = opt_params[0][13:]
                        theta, gammaIH, gammaP, muV, sigma, hospRate, V0, SV0, EV0, P0, \
                            EP0, CP0, CumV0, D0, N, S0, E0, C0, IN0, IH0, RI0, RC0, CumC0, CumIN0, CumIH0, t1V = initializeForVaccineStates(state, t0V, phiV, sigma, hospRate, N, S0, E0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0)
                        (S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV) = ([] for i in range(18))
                        y0 = [S0, V0, SV0, EV0, P0, EP0, E0, CP0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0, CumV0]
                        [x.append(z) for x, z in zip([S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV], y0)]
                        y0 = np.array(y0).flatten()
    
                        for i in range(1, duration + 1):
                            tspan = [t[i - 1], t[i]]
                            # betat = beta
                            if i < 150:
                                betat = beta
                            elif i < 180:
                                betat = beta * (1 + 0.4/30*(i-150))
                            elif i < 330:
                                betat = 1.4 * beta
                            elif i < 366:
                                betat = beta * (1.4 - 0.4/60*(i-330))
    
                            y1 = solve_ivp(lambda t, yy: model(t, yy, theta, thetaV, alphaC, alphaP, betat, gammaC, gammaIN, gammaIH, gammaP, mu, muV, N, sigma, omegaI, omegaC,
                                            omegaV, lambdaV, phiV, eV, t0V, t1V, hospRate), tspan, y0, t_eval=[tspan[1]])
                            y1 = y1.y.flatten()
                            y1 = [y1[i:i + 4] for i in range(0, len(y1), 4)]
                            [x.append(z) for x, z in zip([S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV], y1)]
                            y0 = np.array(y1).flatten()
    
                        IH=pd.DataFrame(IH)
                        IH['all'] = IH.sum(axis=1)
                        output_may21.append([int(eV*100), int(lambdaV*100), vacc_type, t0V, int(1/phiV), int(1/omegaV), list(rho*IH.max(axis=0)), list(D[-1])])

    filepath = './results state vaccine/'+str(int(eV*100))+state+'_May2021.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['efficacy', 'coverage', 'vacc_type', 'vacc_start', 'vacc_window', \
                         'durability', 'hosp', 'death'])
        for row in output_may21:
            writer.writerow(row)


def main():
    state = 'Florida'
    numproc = 8
    pools = mp.Pool(processes=numproc)
    inputList = []
    
    for m in range(numproc):
        inputList.append([state, (m+3)/10]) 
    
    pools.map(runScenarios, inputList) 
    
    output = []
    for m in range(numproc):
        file = './results state vaccine/'+str((m+3)*10)+state+'_May2021.csv'
        with open(file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            for row in reader:
                output.append(row)
        os.remove(file)
    
    output=pd.DataFrame(output)
    output.columns = header
    file = './results state vaccine/'+state+'_May2021.csv'
    output.to_csv(file, index=False)

    
if __name__ == "__main__":
    main()    
    
    