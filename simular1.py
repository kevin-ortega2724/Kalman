import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Constante de tiempo del RTD.
rtd_tau = 60.0

# Pendientes ideales del proceso:
def dT(t):
    return [1/3, 0, 1/3, 0, 1/6, 0, -5/18, -5/18, -5/18, -5/18][int((t-1)/600)]

# Matriz A:
def matrizA(dt, t):
    return np.array([[1, 0, dT(t)*dt], [dt/rtd_tau, 1-dt/rtd_tau, 0], [0, 0, 1]])

# Ruido de proceso: wk ~ N(0, Q)
sigma_wk = 500*0.0005  # 0.25 C, o el 0.05% de la plena escala.
def wk(desv):
    return np.concatenate([norm.rvs(0, desv, size=2), [0]])

# Ruidos de Medida: vk ~ N(0, R)
def outlier(T):
    return T*1.1 if np.random.rand() < 0.0005 else 0

def v_rtd(Ti):
    return np.random.normal(0, 0.01) + np.random.normal(0, 0.005*abs(Ti)) + outlier(Ti)

def v_ntc(Te):
    return np.random.normal(0, 5.5) + outlier(Te)

def v_par(Te):
    return np.random.normal(0, 1.5) + np.random.normal(0, 0.004*abs(Te)) + outlier(Te)

def vk(Te, Ti):
    return np.array([v_rtd(Ti), v_ntc(Te), v_par(Te)])

# Orden de los sensores: RTD, NTC, PAR.
H = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]])

xk = np.zeros((3, 5401))
xk[:, 0] = [0, 0, 1]

zk = np.zeros((3, 5401))
zk[:, 0] = np.dot(H, xk[:, 0])

# Correr el proceso
for t in range(5400):
    # Ecuacion dinamica de Espacio de Estado: x(k+1) = Ax(k) + wk
    xk[:, t+1] = np.dot(matrizA(1, t), xk[:, t]) + wk(sigma_wk)
    # Ecuacion de salida para la simulacion:
    zk[:, t+1] = np.dot(H, xk[:, t+1]) + vk(xk[0, t+1], xk[1, t+1])

# graficar Te y Ti
plt.figure(1)
plt.clf()
plt.plot(range(5401), xk[0, :], '.-')
plt.plot(range(5401), xk[1, :], '.-')
plt.grid(True)
plt.title('Comportamiento ideal del horno CON ruido de proceso')

# graficar T_rtd, T_ntc, T_par
plt.figure(2)
plt.clf()
plt.plot(range(5401), zk[0, :], '.-')
plt.plot(range(5401), zk[1, :], '.-')
plt.plot(range(5401), zk[2, :], '.-')
plt.grid(True)
plt.title('Temperaturas Medidas por los Sensores con ATÍPICOS')

# Simulación de los sensores y el acondicionamiento

# simular el RTD
rtd_Ro = 100
rtd_alpha = 0.0039
R_rtd = rtd_Ro + rtd_alpha*rtd_Ro*zk[0, :]

# simular el acondicionamiento del RTD
rtd_Vcc = 10
rtd_R_lim = 4000
Vm_rtd = (rtd_Vcc*R_rtd)/(rtd_R_lim + R_rtd)
print('Vm_rtd: min=', np.min(Vm_rtd), '\tmax=', np.max(Vm_rtd))  # maximos y minimos
Vdaq_rtd = np.clip(-10 + 20*Vm_rtd, -10, 10)   # DAQ in [-10, +10]

# simular el NTC
ntc_Ro = 821970
ntc_To = 25 + 273.15
ntc_B = 4113
T_ntc = zk[1, :] + 273.15
R_ntc = ntc_Ro * np.exp(ntc_B*((1/T_ntc)-(1/ntc_To)))

# Simular el acondicionamiento del NTC
ntc_R_lim = 10000
ntc_Vcc = 10
Vm_ntc = (ntc_Vcc*R_ntc)/(ntc_R_lim + R_ntc)
print('Vm_ntc: min=', np.min(Vm_ntc), '\tmax=', np.max(Vm_ntc))
Vdaq_ntc = np.clip(-5 + 1.3*Vm_ntc, -10, 10)

# simular el TermoPar. Coeficientes directos.
tipoJ = [0.503811878150E-1,
         0.304758369300E-4,
        -0.856810657200E-7,
         0.132281952950E-9,
        -0.170529583370E-12,
         0.209480906970E-15,
        -0.125383953360E-18,
         0.156317256970E-22]

def E(T):
    return np.dot(tipoJ, [T, T**2, T**3, T**4, T**5, T**6, T**7, T**8])

# Temperatura simulada de la UNION FRIA.
T_amb = 25

# A la Temperatura del termopar se le agrega la union fria (que es como caliente !)
T_par = zk[2, :]
Vm_par = (E(T_par) + E(T_amb)) * 1E-3
print('Vm_par: min=', np.min(Vm_par), '\tmax=', np.max(Vm_par))
Vdaq_par = np.clip(-9 + 500*Vm_par, -10, 10)

# graficar los Vdaq
plt.figure(3)
plt.clf()
plt.plot(Vdaq_rtd)
plt.grid(True)
plt.plot(Vdaq_ntc)
plt.plot(Vdaq_par)
plt.title('Voltajes de Adquisición')
plt.show()
# guardar en archivo:
# usualmente, junto con las señales se guarda la frecuencia de muestreo. En este caso 1 segundo.
fs = 1
# guardar los parametros constantes que se necesiten:
# rtd, ntc, T_amb:
#np.savetxt('senales_daq.txt', Vdaq_rtd=Vdaq_rtd, Vdaq_ntc=Vdaq_ntc, Vdaq_par=Vdaq_par, fs=fs, rtd_Ro=rtd_Ro, rtd_alpha=rtd_alpha, rtd_Vcc=rtd_Vcc, rtd_R_lim=rtd_R_lim, ntc_Ro=ntc_Ro, ntc_To=ntc_To, ntc_B=ntc_B, ntc_R_lim=ntc_R_lim, ntc_Vcc=ntc_Vcc, T_amb=T_amb)
# llamar estimar:
# solucion_t1_estimar

# Convertir valores escalares en arreglos de tamaño 5401
fs_array = np.full_like(Vdaq_rtd, fs)
rtd_Ro_array = np.full_like(Vdaq_rtd, rtd_Ro)
rtd_alpha_array = np.full_like(Vdaq_rtd, rtd_alpha)
rtd_Vcc_array = np.full_like(Vdaq_rtd, rtd_Vcc)
rtd_R_lim_array = np.full_like(Vdaq_rtd, rtd_R_lim)
ntc_Ro_array = np.full_like(Vdaq_rtd, ntc_Ro)
ntc_To_array = np.full_like(Vdaq_rtd, ntc_To)
ntc_B_array = np.full_like(Vdaq_rtd, ntc_B)
ntc_R_lim_array = np.full_like(Vdaq_rtd, ntc_R_lim)
ntc_Vcc_array = np.full_like(Vdaq_rtd, ntc_Vcc)
T_amb_array = np.full_like(Vdaq_rtd, T_amb)

# Combinar los arreglos horizontalmente
combined_data = np.column_stack((Vdaq_rtd, Vdaq_ntc, Vdaq_par, fs_array, rtd_Ro_array, rtd_alpha_array, rtd_Vcc_array, rtd_R_lim_array, ntc_Ro_array, ntc_To_array, ntc_B_array, ntc_R_lim_array, ntc_Vcc_array, T_amb_array))

# Guardar los datos combinados en un archivo .dat
np.save('senales_daq', combined_data)