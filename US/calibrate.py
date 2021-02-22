import random
import numpy as np
from scipy.integrate import odeint
import csv

def calibrate(model, IC, duration, numberCases, numberDeaths, n_iter, filename):
    # IC
    theta0, alpha0, beta0, gammaC0, gammaIN0, gammaIH0, omegaC0, omegaI0, mu0, \
        sigma0, rho0, hospRate0, N0, S0, E0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0 = IC
    
    M = 1e12
    B_list = [0]
    S0_list = [0]
    C0_list = [0]
    IN0_list = [0]
    IH0_list = [0]
    E0_list = [0]
    RI0_list = [0]
    RC0_list = [0]
    CumC0_list = [0]
    CumIN0_list = [0]
    CumIH0_list = [0]
    sigma_list = [0]
    theta_list = [0]
    alpha_list = [0]
    gammaC_list = [0]
    gammaIN_list = [0]
    gammaIH_list = [0]
    omegaC_list = [0]
    omegaI_list = [0]
    mu_list = [0]
    rho_list = [0]
    hospRate_list = [0]
    
    err_list = [M]
    for i in range(n_iter):
        rho0 = random.uniform(0.1, 0.5)
        hospRate0 = random.uniform(0.05, 0.3)
        # theta0 = random.uniform(0.4, 0.8)
        # CumIN0 = (1 - hospRate0) * numberCases[-1] / rho0
        # CumIH0 = hospRate0 * numberCases[-1] / rho0
        CumIN0 = numberCases[0] * (1/rho0 - hospRate0)
        CumIH0 = hospRate0 * numberCases[0]
        CumC0 = (CumIN0 + CumIH0) * (1 - theta0) / theta0
        #
        beta0 = random.uniform(0.1, 0.4)
        E0 = (CumIN0 + CumIH0) * random.uniform(0.01, 0.5)
        RI0 = (CumIN0 + CumIH0 - D0) * random.uniform(0.5, 1)
        RC0 = CumC0 * random.uniform(0.5, 1)
        sigma0 = random.uniform(0, 0.05)
        # alpha0 = random.uniform(0.4, 0.99)
        # gammaC0 = random.uniform(1 / 10, 1 / 3)
        # gammaIN0 = random.uniform(1 / 10, 1 / 3)
        # gammaIH0 = random.uniform(1 / 10, 1 / 3)
        # omegaC0 = random.uniform(0, 1/60)
        # omegaI0 = min(omegaC0, random.uniform(0, 1/90))
        # mu0 = random.uniform(1 / 10, 1/2)
        #
        IN0 = (1 - hospRate0) * (CumIN0 + CumIH0 - D0 - RI0)
        IH0 = hospRate0 * (CumIN0 + CumIH0 - D0 - RI0)
        C0 = CumC0 - RC0
        S0 = N0 - E0 - C0 - IN0 - IH0 - RC0 - RI0 - D0
    
        t = np.linspace(0, duration, duration + 1)
        (S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH) = ([] for i in range(11))
        y0 = [S0, E0, C0, IN0, IH0, RI0, RC0, D0, CumC0, CumIN0, CumIH0]
        [x.append(z) for x, z in zip([S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH], y0)]
        theta, alpha, beta, gammaC, gammaIN, gammaIH, mu, N, sigma, omegaI, omegaC, hospRate = [theta0, alpha0, beta0,
                                                                                     gammaC0, gammaIN0, gammaIH0, 
                                                                                     mu0, N0, sigma0, omegaI0, omegaC0, hospRate0]
        for i in range(1, duration + 1):
            tspan = [t[i - 1], t[i]]
            y = odeint(model, y0, tspan, args=(theta, alpha, beta, gammaC, gammaIN, gammaIH, mu, N, sigma, omegaI, omegaC, hospRate))
            [x.append(z) for x, z in zip([S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH], y[1])]
            y0 = y[1]
    
        # Isource = state_data[start_date_ind:end_date_ind + 1]
        # Dsource = state_death[start_date_ind:end_date_ind + 1]
        err = [((numberCases[i] - (CumIN[i] + CumIH[i]) * rho0) ** 2) / 1e9 + ((numberDeaths[i] - D[i]) ** 2) / 1e9*0 for i in range(len(CumIN))]
        err = sum(err)
        if err < max(err_list):
            if len(err_list) < 10:
                B_list.append(beta0)
                C0_list.append(C0)
                E0_list.append(E0)
                IN0_list.append(IN0)
                IH0_list.append(IH0)
                S0_list.append(S0)
                RI0_list.append(RI0)
                RC0_list.append(RC0)
                CumC0_list.append(CumC0)
                CumIN0_list.append(CumIN0)
                CumIH0_list.append(CumIH0)
                err_list.append(err)
                sigma_list.append(sigma0)
                theta_list.append(theta0)
                alpha_list.append(alpha0)
                gammaC_list.append(gammaC0)
                gammaIN_list.append(gammaIN0)
                gammaIH_list.append(gammaIH0)
                omegaC_list.append(omegaC0)
                omegaI_list.append(omegaI0)
                mu_list.append(mu0)
                rho_list.append(rho0)
                hospRate_list.append(hospRate0)
            else:
                ind = err_list.index(max(err_list))
                B_list[ind] = beta0
                C0_list[ind] = C0
                E0_list[ind] = E0
                IN0_list[ind] = IN0
                IH0_list[ind] = IH0
                S0_list[ind] = S0
                RI0_list[ind] = RI0
                RC0_list[ind] = RC0
                CumC0_list[ind] = CumC0
                CumIN0_list[ind] = CumIN0
                CumIH0_list[ind] = CumIH0
                err_list[ind] = err
                sigma_list[ind] = sigma0
                theta_list[ind] = theta0
                alpha_list[ind] = alpha0
                gammaC_list[ind] = gammaC0
                gammaIN_list[ind] = gammaIN0
                gammaIH_list[ind] = gammaIH0
                omegaC_list[ind] = omegaC0
                omegaI_list[ind] = omegaI0
                mu_list[ind] = mu0
                rho_list[ind] = rho0
                hospRate_list[ind] = hospRate0
    
    ind = np.argsort(err_list)
    err_sorted = np.sort(err_list)
    beta_sorted = [B_list[i] for i in ind]
    C0_sorted = [C0_list[i] for i in ind]
    E0_sorted = [E0_list[i] for i in ind]
    IN0_sorted = [IN0_list[i] for i in ind]
    IH0_sorted = [IH0_list[i] for i in ind]
    S0_sorted = [S0_list[i] for i in ind]
    RI0_sorted = [RI0_list[i] for i in ind]
    RC0_sorted = [RC0_list[i] for i in ind]
    CumC0_sorted = [CumC0_list[i] for i in ind]
    CumIN0_sorted = [CumIN0_list[i] for i in ind]
    CumIH0_sorted = [CumIH0_list[i] for i in ind]
    sigma_sorted = [sigma_list[i] for i in ind]
    theta_sorted = [theta_list[i] for i in ind]
    alpha_sorted = [alpha_list[i] for i in ind]
    gammaC_sorted = [gammaC_list[i] for i in ind]
    gammaIN_sorted = [gammaIN_list[i] for i in ind]
    gammaIH_sorted = [gammaIH_list[i] for i in ind]
    omegaC_sorted = [omegaC_list[i] for i in ind]
    omegaI_sorted = [omegaI_list[i] for i in ind]
    mu_sorted = [mu_list[i] for i in ind]
    rho_sorted = [rho_list[i] for i in ind]
    hospRate_sorted = [hospRate_list[i] for i in ind]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['err', 'N', 'S0', 'E0', 'C0', 'IN0', 'IH0', 'RI0', 'RC0', 'CumC0', 
                         'CumIN0', 'CumIH0', 'D0', 'theta', 'alpha', 'beta', 'gammaC', 'gammaIN', 'gammaIH',
                         'omegaC', 'omegaI', 'mu', 'sigma', 'rho', 'hospRate'])
        for row in zip(err_sorted, N0, S0_sorted, E0_sorted, C0_sorted, IN0_sorted, IH0_sorted, 
                       RI0_sorted, RC0_sorted, CumC0_sorted, CumIN0_sorted, CumIH0_sorted, D0,
                       theta_sorted, alpha_sorted, beta_sorted, gammaC_sorted, gammaIN_sorted, 
                       gammaIH_sorted, omegaC_sorted, omegaI_sorted, mu_sorted,
                       sigma_sorted, rho_sorted, hospRate_sorted):
            writer.writerow(row)
