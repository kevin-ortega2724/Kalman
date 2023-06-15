import numpy as np
import matplotlib.pyplot as plt

# Cargar paquetes
from scipy import stats

# SIMULAR QUE SE LEEN DATOS DEL SISTEMA DE ADQUISICIÓN
senales_daq = np.load('senales_daq.npy')

# Acceder a los elementos del arreglo
Vdaq_par = senales_daq[:, 2]
T_amb = senales_daq[:, 13]
# Convertir los voltajes leídos por Temperaturas
tipoJ = [0.503811878150E-1, 0.304758369300E-4, -0.856810657200E-7, 0.132281952950E-9,
         -0.170529583370E-12, 0.209480906970E-15, -0.125383953360E-18, 0.156317256970E-22]

E = lambda T: np.dot(tipoJ, [T, T**2, T**3, T**4, T**5, T**6, T**7, T**8])

invsJ = [1.978425E1, -2.001204E-1, 1.036969E-2, -2.549687E-4, 3.585153E-6, -5.344285E-8, 5.099890E-10]

T90 = lambda E: np.dot(invsJ, [E, E**2, E**3, E**4, E**5, E**6, E**7])

Vm_par = (1 / 500) * (senales_daq[:, 2] + 9) * 1E3 - E(T_amb)
T_par = T90(Vm_par)

# Definición de las constantes de NTC
ntc = {
    'Ro': 10.0,   # Valor de resistencia a la temperatura de referencia
    'To': 25.0,   # Temperatura de referencia en grados Celsius
    'B': 3950.0,  # Coeficiente B de la curva característica
    'R_lim': 1.0, # Límite de resistencia del NTC
    'Vcc': 5.0    # Tensión de alimentación del NTC
}

Vm_ntc = (1 / 1.3) * (senales_daq[:, 1] + 5)
R_ntc = (Vm_ntc * ntc['R_lim']) / (ntc['Vcc'] - Vm_ntc)
T_ntc = 1 / ((1 / ntc['B']) * np.log(R_ntc / ntc['Ro']) + (1 / ntc['To'])) - 273.15


Vm_rtd = (1 / 20) * (senales_daq[:, 0] + 10)
R_rtd = (Vm_rtd * rtd.R_lim) / (rtd.Vcc - Vm_rtd)
T_rtd = (1 / (rtd.Ro * rtd.alpha)) * (R_rtd - rtd.Ro)

plt.figure(11)
plt.clf()
plt.plot(T_par)
plt.plot(T_ntc)
plt.plot(T_rtd)
plt.legend(['T TERMOCUPLA', 'T NTC', 'T RTD'])
plt.title('Temperaturas Recuperadas de la Adquisición')

# Estimador de Kalman
xk = np.zeros((3, 5401))
xk[:, 0] = [0, 0, 1]
Pk = np.zeros((3, 3, 5401))
Pk[:, :, 0] = np.eye(3)

sigma_wk = 0.5
Q = lambda dt: np.array([[dt**2, 0, 0], [0, dt**2, 0], [0, 0, 0]]) * sigma_wk**2

R = np.array([3.0, 5.5, 3.0])**2

dT = lambda t: np.array([1/3, 0, 1/3, 0, 1/6, 0, -5/18, -5/18, -5/18, -5/18])[int(t/600)]
matrizA = lambda dt, t: np.array([[1, 0, dT(t)*dt], [dt/rtd.tau, 1-dt/rtd.tau, 0], [0, 0, 1]])
Hcomb = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]])

last_t = 0
for t in range(5400):
    A = matrizA(t - last_t, t)
    xk[:, t+1] = np.dot(A, xk[:, last_t+1])
    Pk[:, :, t+1] = np.dot(np.dot(A, Pk[:, :, last_t+1]), A.T) + Q(t - last_t)

    sensores = [T_rtd[t], T_ntc[t], T_par[t]]
    for i in range(3):
        zk = sensores[i]
        zscore = np.abs(zk - np.dot(Hcomb[i, :], xk[:, t+1])) / np.sqrt(Pk[1, 1, t+1, i])
        if zscore < 3:
            H = Hcomb[i, :]
            Kk = np.dot(np.dot(Pk[:, :, t+1], H.T), np.linalg.inv(np.dot(np.dot(H, Pk[:, :, t+1]), H.T) + R[i]))
            xk[:, t+1] = xk[:, t+1] + np.dot(Kk, (zk - np.dot(H, xk[:, t+1])))
            Pk[:, :, t+1] = np.dot((np.eye(3) - np.dot(Kk, H)), Pk[:, :, t+1])
            last_t = t

plt.figure(12)
plt.clf()
plt.plot(xk[0:2, :].T, linewidth=1)
plt.grid(True)
plt.legend(['Te', 'Ti'])
plt.title('Resultado de la Estimación')

sigma_te = np.sqrt(np.reshape(Pk[0, 0, :], (1, 5401)))
plt.plot(xk[0, :] + 4 * sigma_te, 'k--')
plt.plot(xk[0, :] - 4 * sigma_te, 'k--')
