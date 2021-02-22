import numpy as np

def initializeForCalibration(state, numberCases, numberDeath):
    theta = 0.6
    alpha, beta = [0.75, 0.2]
    gammaC, gammaIN, gammaIH = [1/10, 1/10, 1/8]
    mu = 1/6
    sigma = 0.05/15
    omegaI, omegaC = [1/270, 1/90]
    rho = 1/5
    hospRate = 0.1
    # n_cases = state_data[start_date_ind]
    # n_death = state_death[start_date_ind]    
    if state == 'Maryland':
        N = 6e6
    elif state == 'Wisconsin':
        N = 5.8e6
    elif state == 'Florida':
        N = 21.5e6
    #
    RI0 = 0
    RC0 = 0
    exposure_rate = 0.1  # estimated number of exposed to number of cumulative infected
    CumIN0 = (1 - hospRate) * numberCases / rho
    CumIH0 = hospRate * numberCases / rho
    D0 = numberDeath
    CumC0 = (CumIN0 + CumIH0) * (1 - theta) / theta
    IN0 = (1 - hospRate) * (CumIN0 + CumIH0 - D0 - RI0)
    IH0 = hospRate * (CumIN0 + CumIH0 - D0 - RI0)
    E0 = (CumIN0 + CumIH0) * exposure_rate
    C0 = CumC0 - RC0
    S0 = N - E0 - CumC0 - (CumIN0 + CumIH0) - D0
    
    return (theta, alpha, beta, gammaC, gammaIN, gammaIH, omegaC, omegaI, mu, 
              sigma, rho, hospRate, N, S0, E0, C0, IN0, IH0, RI0, RC0, D0, 
              CumC0, CumIN0, CumIH0)

def initializeForVaccineScenarios(t0V, phiV, aggregate=False):
    popRatio = np.array([0.24, 0.37, 0.26, 0.13])
    theta = np.array([0.21, 0.5, 0.5, 0.6])
    alphaC, beta = [0.75, 0.25] #[0.25, 0.35]
    gammaC, gammaIN = [1/10, 1/10]
    gammaIH = np.array([1/7, 1/7, 1/9, 1/9])
    gammaP = 1/4
    mu, muV = [1/6, 1/21]
    N = 100000 * popRatio
    mortality = np.array([0, 0.024, 0.1, 0.266])
    sigma_hat = np.array([1/15, 1/15, 1/17, 1/13])
    sigma = mortality * sigma_hat
    hospRate = np.array([0.05, 0.18, 0.2, 0.2])
    omegaI, omegaC = [1/270, 1/90]
    V0 = np.zeros(len(popRatio))
    SV0 = np.zeros(len(popRatio))
    EV0 = np.zeros(len(popRatio))
    P0 = np.zeros(len(popRatio))
    EP0 = np.zeros(len(popRatio))
    CP0 = np.zeros(len(popRatio))
    RI0 = np.zeros(len(popRatio))
    RC0 = np.zeros(len(popRatio))
    CumV0 = np.zeros(len(popRatio))
    n_cases = 0.0001 * sum(N) * popRatio
    n_death = 0 * sum(N) * popRatio
    exposure_rate = 1  # estimated number of exposed to number of cumulative infected at t=0
    #
    t1V = t0V + 1 / phiV
    CumIN0 = (1 - hospRate) * n_cases
    CumIH0 = hospRate * n_cases
    D0 = n_death
    CumC0 = (CumIN0 + CumIH0)*(1-theta)/theta
    IN0 = (1 - hospRate) * (CumIN0 + CumIH0 - D0 - RI0)
    IH0 = hospRate * (CumIN0 + CumIH0 - D0 - RI0)
    E0 = (CumIN0 + CumIH0) * exposure_rate
    C0 = CumC0 - RC0
    S0 = N - E0 - CumC0 - (CumIN0 + CumIH0) - D0
    
    if aggregate:
        theta, gammaIH, sigma, hospRate = list(map(lambda x: sum(popRatio*x), [theta, gammaIH, sigma, hospRate]))
        
    
    return (theta, alphaC, beta, gammaC, gammaIN, gammaIH, gammaP, omegaC, omegaI, mu, 
              muV, sigma, hospRate, N, t1V, S0, V0, SV0, EV0, P0, EP0, E0,
              CP0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0, CumV0)
    

def initializeForVaccineStates(state, t0V, phiV, sigma, hospRate, N, S0, E0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0):
    if state = 'Maryland':
        popRatio = np.array([0.28, 0.45, 0.16, 0.11])
    elif state = 'Wisconsin':
        popRatio = np.array([0.29, 0.43, 0.15, 0.13])
    elif state = 'Florida':
        popRatio = np.array([0.25, 0.41, 0.16, 0.18])
        
    theta = np.array([0.21, 0.5, 0.5, 0.6])
    gammaIH = np.array([1/7, 1/7, 1/9 , 1/9])
    gammaP = 1/4
    muV = 1/21
    mortality = np.array([0, 0.024, 0.1, 0.266])
    sigma_hat = np.array([1/15, 1/15, 1/17, 1/13])
    sigma0 = mortality * sigma_hat
    sigma0 = sigma0 / sum(popRatio*sigma0) * sigma
    hospRate0 = np.array([0.05, 0.18, 0.2, 0.2])
    hospRate0 = hospRate0 / sum(popRatio*hospRate0) * hospRate
    V0 = np.zeros(len(popRatio))
    SV0 = np.zeros(len(popRatio))
    EV0 = np.zeros(len(popRatio))
    P0 = np.zeros(len(popRatio))
    EP0 = np.zeros(len(popRatio))
    CP0 = np.zeros(len(popRatio))
    CumV0 = np.zeros(len(popRatio))
    D0 = D0 * mortality / sum(mortality)
    N, S0, E0, C0, IN0, IH0, RI0, RC0, CumC0, CumIN0, CumIH0 = list(map(lambda x: x*popRatio, [N, S0, E0, C0, IN0, IH0, RI0, RC0, CumC0, CumIN0, CumIH0]))
    t1V = t0V + 1 / phiV
    
    return (theta, gammaIH, gammaP, muV, sigma0, hospRate0, V0, SV0, EV0, P0, 
            EP0, CP0, CumV0, D0, N, S0, E0, C0, IN0, IH0, RI0, RC0, CumC0, CumIN0, CumIH0, t1V)