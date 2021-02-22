import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar
import random
import time
import math
import os
from datetime import datetime
from datetime import timedelta
from scipy.integrate import odeint
from EpiModels import reinfection_model_hospital
from calibrate import calibrate
from reset_parameters import initializeForCalibration
import multiprocessing as mp

def mp_calibrate(inputList):
    state, model, IC, duration, numberCases, numberDeaths, i = inputList
    filename = str(i)+state+'_since_may_fixed_params.csv'
    calibrate(model, IC, duration, numberCases, numberDeaths, 10000, filename)

def main():
    # Select model
    model = reinfection_model_hospital

    filepath = 'time_series_covid19_confirmed_US.csv'
    state = 'Florida'
    with open(filepath, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))

    filepath = 'time_series_covid19_deaths_US.csv'
    with open(filepath, newline='') as csvfile:
        Ddata = list(csv.reader(csvfile, delimiter=','))

    county_data = []
    county_death = []
    county_names = []
    dates = data[0][11:]
    for i in range(1, len(data)):  # data[1:]:
        if data[i][6] == state:
            county_names.append(data[i][5])
            county_data.append([int(j) for j in data[i][11:]])
            county_death.append([int(j) for j in Ddata[i][12:]])

    n_days = len(county_data[0])
    state_data = [sum(x) for x in zip(*county_data)]
    state_death = [sum(x) for x in zip(*county_death)]

    # plt.figure(1)
    # plt.plot(range(n_days), state_data)
    # plt.figure(2)
    # plt.plot(range(n_days-1), np.diff(state_data))

    start_date = '5/1/20'
    end_date = '10/1/20'
    start_date_ind = dates.index(start_date)
    end_date_ind = dates.index(end_date)
    start_date = datetime.strptime(start_date, '%m/%d/%y')
    end_date = datetime.strptime(end_date, '%m/%d/%y')
    duration = end_date - start_date
    duration = duration.days
    t = np.linspace(0, duration, duration + 1)
    numberCases = state_data[start_date_ind:end_date_ind + 1]
    numberDeaths = state_death[start_date_ind:end_date_ind + 1]

    # Calibration
    filename = state+'_since_may_fixed_params.csv'
    IC = initializeForCalibration(state, numberCases[0], numberDeaths[0])
    theta, alpha, beta, gammaC, gammaIN, gammaIH, omegaC, omegaI, mu, sigma, rho, \
        hospRate, N, S0, E0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0 = IC
        
    numproc = mp.cpu_count()
    pools = mp.Pool(processes=numproc)
    inputList = []
    
    for m in range(numproc):
        inputList.append([state, model, IC, duration, numberCases, numberDeaths, m]) 
    
    pools.map(mp_calibrate, inputList) 
    opt_params = []
    for m in range(numproc):
        file = str(m)+state+'_since_may_fixed_params.csv'
        with open(file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            for row in reader:
                opt_params.append([float(el) for el in row])
        os.remove(file)
    
    opt_params=pd.DataFrame(opt_params)
    opt_params = opt_params.sort_values(by=[0]) 
    opt_params = opt_params.iloc[0:9]
    opt_params.columns = header
    opt_params.to_csv(filename, index=False)
    
    # run with optimal parameters
    alt = 0
    opt_params = np.array(opt_params.iloc[alt])
    
    S0, E0, C0, IN0, IH0, RI0, RC0, CumC0, CumIN0, CumIH0 = opt_params[1:11]
    (S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH) = ([] for i in range(11))
    y0 = [S0, E0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0]
    [x.append(z) for x, z in zip([S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH], y0)]
    theta, alpha, beta, gammaC, gammaIN, gammaIH, omegaC, omegaI, mu, sigma, rho, hospRate = opt_params[11:]
    duration2 = duration
    t2 = np.linspace(0, duration2, duration2 + 1)

    for i in range(1, duration2 + 1):
        tspan = [t2[i-1], t2[i]]
        y = odeint(model, y0, tspan, args=(theta, alpha, beta, gammaC, gammaIN, gammaIH, mu, N, sigma, omegaI, omegaC, hospRate))
        [x.append(z) for x, z in zip([S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH], y[1])]
        y0 = y[1]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.2))
    ax[0].plot(t[1:], 0.001*np.diff(state_data[start_date_ind:end_date_ind+1]), label="Data")
    ax[0].plot(t2[1:], 0.001*np.diff([rho*(CumIN[i]+CumIH[i]) for i in range(len(CumIN))]), label='Model')
    ax[0].set_ylabel('Daily New Cases, thousands', fontsize=20)
    ax[0].set_xticks(range(0, duration2, 30))
    ax[0].set_xticklabels([x.strftime('%m/%y') for x in
                         [start_date + timedelta(days=31 * i) for i in
                          range(math.ceil(duration2 / 30))]], rotation='vertical', fontsize=20)
    ax[0].tick_params(labelsize=20)

    ax[1].plot(t, 0.001*np.array(state_death[start_date_ind:end_date_ind+1]), label="Data")
    ax[1].plot(t2, 0.001*np.array(D), label='Model')
    ax[1].set_ylabel('Death Cases, thousands', fontsize=20)
    ax[1].set_xticks(range(0, duration2, 30))
    ax[1].set_xticklabels([x.strftime('%m/%y') for x in
                         [start_date + timedelta(days=31 * i) for i in
                          range(math.ceil(duration2 / 30))]], rotation='vertical', fontsize=20)
    ax[1].tick_params(labelsize=20)
    lgd = ax[1].legend(loc="lower left", bbox_to_anchor=(-0.5, -0.35), ncol=5, fontsize=20)
    plt.savefig(state+'_fitted.png', bbox_extra_artists=(lgd,), bbox_inches='tight')            
                

if __name__ == "__main__":
    main()