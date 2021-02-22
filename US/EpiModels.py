import numpy as np


def reinfection_model(y, t, theta, alpha, beta, gammaC, gammaI, mu, N, sigma, omegaI, omegaC):
    (S, E, C, I, RI, RC, D, CumC, CumI) = y

    S_dot = -beta * S * (alpha * C + I) / N + omegaI * RI + omegaC * RC
    E_dot = beta * S * (alpha * C + I) / N - mu * E
    C_dot = mu * (1 - theta) * E - gammaC * C
    I_dot = mu * theta * E - (gammaI + sigma) * I
    RI_dot = gammaI * I - omegaI * RI
    RC_dot = gammaC * C - omegaC * RC
    D_dot = sigma * I
    CumC_dot = mu * (1 - theta) * E
    CumI_dot = mu * theta * E

    dy = [S_dot, E_dot, C_dot, I_dot, RI_dot, RC_dot, D_dot, CumC_dot, CumI_dot]
    return dy

def reinfection_model_hospital(y, t, theta, alpha, beta, gammaC, gammaIN, gammaIH, mu, N, sigma, omegaI, omegaC, hospRate):
    (S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH) = y

    S_dot = -beta * S * (alpha * C + IN + IH) / N + omegaI * RI + omegaC * RC
    E_dot = beta * S * (alpha * C + IN + IH) / N - mu * E
    C_dot = mu * (1 - theta) * E - gammaC * C
    IN_dot = (1 - hospRate) * mu * theta * E - gammaIN * IN
    IH_dot = hospRate * mu * theta * E - (gammaIH + sigma) * IH
    RI_dot = gammaIN * IN + gammaIH * IH - omegaI * RI
    RC_dot = gammaC * C - omegaC * RC
    D_dot = sigma * IH
    CumC_dot = mu * (1 - theta) * E
    CumIN_dot = (1 - hospRate) * mu * theta * E
    CumIH_dot = hospRate * mu * theta * E

    dy = [S_dot, E_dot, C_dot, IN_dot, IH_dot, RI_dot, RC_dot, D_dot, CumC_dot, CumIN_dot, CumIH_dot]
    return dy

def reinfection_model_hospital_age(y, t, theta, alpha, beta, gammaC, gammaIN, gammaIH, mu, N, sigma, omegaI, omegaC, hospRate):
    (S, E, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH) = [y[i:i + 4] for i in range(0, len(y), 4)]

    S_dot = -beta * S * (alpha * sum(C) + sum(IN) + sum(IH)) / sum(N) + omegaI * RI + omegaC * RC
    E_dot = beta * S * (alpha * sum(C) + sum(IN) + sum(IH)) / sum(N) - mu * E
    C_dot = mu * (1 - theta) * E - gammaC * C
    IN_dot = (1 - hospRate) * mu * theta * E - gammaIN * IN
    IH_dot = hospRate * mu * theta * E - (gammaIH + sigma) * IH
    RI_dot = gammaIN * IN + gammaIH * IH - omegaI * RI
    RC_dot = gammaC * C - omegaC * RC
    D_dot = sigma * IH
    CumC_dot = mu * (1 - theta) * E
    CumIN_dot = (1 - hospRate) * mu * theta * E
    CumIH_dot = hospRate * mu * theta * E

    dy = [*S_dot, *E_dot, *C_dot, *IN_dot, *IH_dot, *RI_dot, *RC_dot, *D_dot, 
          *CumC_dot, *CumIN_dot, *CumIH_dot]
    return dy


def SECIR_vacc_model(t, y, theta, thetaV, alphaC, alphaP, beta, gammaC, gammaI, gammaP, mu, muV, N, sigma, omegaI, omegaC,
                    omegaV, lambdaV, phiV, xiV, eV, t0V, t1V):
    (S, V, SV, EV, P, EP, E, CP, CV, C, I, RI, RC, RCV, D, CumC, CumI, CumV) = y

    S_dot = -beta * S * (alphaC * C + I + alphaP * CP + alphaC * CV) / N + np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (
                -lambdaV * phiV * xiV * S) + omegaC * RC + omegaI * RI + omegaC * SV + omegaV * P
    V_dot = np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (
                lambdaV * phiV * xiV * S) - muV * V - beta * V * (alphaC * C + I + alphaP * CP + alphaC * CV) / N
    SV_dot = (1-eV) * muV * V - beta * SV * (alphaC * C + I + alphaP * CP + alphaC * CV) / N - omegaC * SV
    EV_dot = beta * SV * (alphaC * C + I + alphaP * CP + alphaC * CV) / N - mu * EV
    P_dot = eV * muV * V + gammaP * CP - beta * P * (alphaC * C + I + alphaP * CP + alphaC * CV) / N - omegaV * P
    EP_dot = beta * P * (alphaC * C + I + alphaP * CP + alphaC * CV) / N - mu * EP
    E_dot = beta * (S + V) * (alphaC * C + I + alphaP * CP + alphaC * CV) / N - mu * E
    CP_dot = mu * EP - gammaP * CP
    CV_dot = 0
    C_dot = (1 - theta) * mu * E + (1-thetaV) * mu * EV - gammaC * C
    I_dot = theta * mu * E + thetaV * mu * EV - (gammaI + sigma) * I
    RI_dot = gammaI * I - omegaI * RI
    RC_dot = gammaC * C - omegaC * RC
    D_dot = sigma * I
    CumC_dot = (1 - theta) * mu * E + (1-thetaV) * mu * EV
    CumI_dot = theta * mu * E + thetaV * mu * EV
    CumV_dot = np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (lambdaV * phiV * xiV * S)
    RCV_dot = 0
    dy = [S_dot, V_dot, SV_dot, EV_dot, P_dot, EP_dot, E_dot, CP_dot, C_dot, 
          I_dot, RI_dot, RC_dot, D_dot, CumC_dot, CumI_dot, CumV_dot]
    return dy

def SECIR_vacc_model_hospital(t, y, theta, thetaV, alphaC, alphaP, beta, gammaC, gammaIN, gammaIH, gammaP, mu, muV, N, sigma, omegaI, omegaC,
                    omegaV, lambdaV, phiV, eV, t0V, t1V, hospRate):
    (S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV) = y

    S_dot = -beta * S * (alphaC * C + IN + IH + alphaP * CP) / N + np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (
                -lambdaV * phiV * S) + omegaC * RC + omegaI * RI + omegaC * SV + omegaV * P
    V_dot = np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (
                lambdaV * phiV * S) - muV * V - beta * V * (alphaC * C + IN + IH + alphaP * CP) / N
    SV_dot = (1-eV) * muV * V - beta * SV * (alphaC * C + IN + IH + alphaP * CP) / N - omegaC * SV
    EV_dot = beta * SV * (alphaC * C + IN + IH + alphaP * CP) / N - mu * EV
    P_dot = eV * muV * V + gammaP * CP - beta * P * (alphaC * C + IN + IH + alphaP * CP) / N - omegaV * P
    EP_dot = beta * P * (alphaC * C + IN + IH + alphaP * CP) / N - mu * EP
    E_dot = beta * (S + V) * (alphaC * C + IN + IH + alphaP * CP) / N - mu * E
    CP_dot = mu * EP - gammaP * CP
    C_dot = (1 - theta) * mu * E + (1-thetaV) * mu * EV - gammaC * C
    IN_dot = (1 - hospRate) * (theta * mu * E + thetaV * mu * EV) - gammaIN * IN
    IH_dot = hospRate * (theta * mu * E + thetaV * mu * EV) - (gammaIH + sigma) * IH
    RI_dot = gammaIN * IN + gammaIH * IH - omegaI * RI
    RC_dot = gammaC * C - omegaC * RC
    D_dot = sigma * IH
    CumC_dot = (1 - theta) * mu * E + (1-thetaV) * mu * EV
    CumIN_dot = (1 - hospRate) * (theta * mu * E + thetaV * mu * EV)
    CumIH_dot = hospRate * (theta * mu * E + thetaV * mu * EV)
    CumV_dot = np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (lambdaV * phiV * S)
    dy = [S_dot, V_dot, SV_dot, EV_dot, P_dot, EP_dot, E_dot, CP_dot, C_dot, IN_dot, IH_dot, RI_dot, RC_dot,
          D_dot, CumC_dot, CumIN_dot, CumIH_dot, CumV_dot]
    return dy

def SECIR_vacc_model_hospital_age(t, y, theta, thetaV, alphaC, alphaP, beta, gammaC, gammaIN, gammaIH, gammaP, mu, muV, N, sigma, omegaI, omegaC,
                    omegaV, lambdaV, phiV, eV, t0V, t1V, hospRate):
    (S, V, SV, EV, P, EP, E, CP, C, IN, IH, RI, RC, D, CumC, CumIN, CumIH, CumV) = [y[i:i + 4] for i in range(0, len(y), 4)]

    S_dot = -beta * S * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N) + np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (
                -lambdaV * phiV * S) + omegaC * RC + omegaI * RI + omegaC * SV + omegaV * P
    V_dot = np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (
                lambdaV * phiV * S) - muV * V - beta * V * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N)
    SV_dot = (1-eV) * muV * V - beta * SV * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N) - omegaC * SV
    EV_dot = beta * SV * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N) - mu * EV
    P_dot = eV * muV * V + gammaP * CP - beta * P * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N) - omegaV * P
    EP_dot = beta * P * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N) - mu * EP
    E_dot = beta * (S + V) * (alphaC * sum(C) + sum(IN) + sum(IH) + alphaP * sum(CP)) / sum(N) - mu * E
    CP_dot = mu * EP - gammaP * CP
    C_dot = (1 - theta) * mu * E + (1-thetaV) * mu * EV - gammaC * C
    IN_dot = (1 - hospRate) * (theta * mu * E + thetaV * mu * EV) - gammaIN * IN
    IH_dot = hospRate * (theta * mu * E + thetaV * mu * EV) - (gammaIH + sigma) * IH
    RI_dot = gammaIN * IN + gammaIH * IH - omegaI * RI
    RC_dot = gammaC * C - omegaC * RC
    D_dot = sigma * IH
    CumC_dot = (1 - theta) * mu * E + (1-thetaV) * mu * EV
    CumIN_dot = (1 - hospRate) * (theta * mu * E + thetaV * mu * EV)
    CumIH_dot = hospRate * (theta * mu * E + thetaV * mu * EV)
    CumV_dot = np.heaviside(t - t0V, 1) * np.heaviside(t1V - t, 1) * (lambdaV * phiV * S)
    dy = [*S_dot, *V_dot, *SV_dot, *EV_dot, *P_dot, *EP_dot, *E_dot, *CP_dot, 
          *C_dot, *IN_dot, *IH_dot, *RI_dot, *RC_dot, *D_dot, *CumC_dot, 
          *CumIN_dot, *CumIH_dot, *CumV_dot]
    return dy
