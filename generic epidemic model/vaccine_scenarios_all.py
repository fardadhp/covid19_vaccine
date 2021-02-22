import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar
import random
import time
from datetime import datetime
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from EpiModels import SECIR_vacc_model_hospital_age
from reset_parameters import initializeForVaccineScenarios


# Select model
model = SECIR_vacc_model_hospital_age
duration = 365
path = "results vaccine scenarios\\high alpha\\all\\"
t = np.linspace(0, duration, duration + 1)
theta, alphaC, beta, gammaC, gammaIN, gammaIH, gammaP, omegaC, omegaI, mu, \
    muV, sigma, hospRate, N, t1V, S0, V0, SV0, EV0, P0, EP0, E0, CP0, C0, IN0, \
        IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0, CumV0 = initializeForVaccineScenarios(0, 1/60, aggregate=False)

for eV in np.arange(0.1,1.01,0.1):
    for lambdaV in np.arange(0.1,1.01,0.1):
        for t0V in np.arange(0,7)*30:
            output = []
            for phiV in [1/30, 1/60, 1/90, 1/120]:
                for omegaV in [1/90, 1/180, 1/270, 1/360]:
                    for sc in range(4):
                        if sc == 0:
                            lbl = 'I0-S0'
                            alphaP = 0
                            thetaV = 0
                        elif sc == 1:
                            lbl = 'I1-S0'
                            alphaP = alphaC
                            thetaV = 0
                        elif sc == 2:
                            lbl = 'I0-S1'
                            alphaP = 0
                            thetaV = theta
                        elif sc == 3:
                            lbl = 'I1-S1'
                            alphaP = alphaC
                            thetaV = theta
            
                        theta, alphaC, beta, gammaC, gammaIN, gammaIH, gammaP, omegaC, omegaI, mu, \
                            muV, sigma, hospRate, N, t1V, S0, V0, SV0, EV0, P0, EP0, E0, \
                                CP0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0, CumV0 = initializeForVaccineScenarios(t0V, phiV, aggregate=False)
                        (S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV) = ([] for i in range(18))
                        y0 = [S0, V0, SV0, EV0, P0, EP0, E0, CP0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0, CumV0]
                        [x.append(z) for x, z in zip([S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV], y0)]
                        y0 = np.array(y0).flatten()
                        
                        for i in range(1, duration + 1):
                            tspan = [t[i-1], t[i]]
                            y1 = solve_ivp(lambda t, yy: model(t, yy, theta, thetaV, alphaC, alphaP, beta, gammaC, gammaIN, gammaIH, gammaP, mu, muV, N, sigma, omegaI, omegaC,
                                        omegaV, lambdaV, phiV, eV, t0V, t1V, hospRate), tspan, y0, t_eval=[tspan[1]])
                            y1 = y1.y.flatten()
                            y1 = [y1[i:i + 4] for i in range(0, len(y1), 4)]
                            [x.append(z) for x, z in zip([S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV], y1)]
                            y0 = np.array(y1).flatten()
                        
                        IH=pd.DataFrame(IH)
                        IH['all'] = IH.sum(axis=1)
                        output.append([int(eV*100), int(lambdaV*100), int(t0V/30), int(1/phiV), int(1/omegaV), lbl, list(IH.max(axis=0)), list(D[-1])])

            with open(path+'vaccine_scenarios_'+str(int(eV*100))+ '_' + \
                      str(int(lambdaV*100)) + '_' + str(int(t0V/30)) + '.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['efficacy', 'coverage', 'vacc_start', 'vacc_window', \
                                 'durability', 'vacc_type', 'hosp', 'death'])
                for row in output:
                    writer.writerow(row)

