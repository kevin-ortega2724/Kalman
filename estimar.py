# Cargar paquetes
import statistics
import numpy as np
import matplotlib.pyplot as plt

# SIMULAR QUE SE LEEN DATOS DEL SISTEMA DE ADQUISICIÓN.
senales_daq = np.loadtxt('senales_daq.dat')

# Convertir los voltajes leidos por Temperaturas, de acuerdo con el conocimiento
# que se tiene del sistema de acondicionamiento de señal.
# Por cada sensor, es una sola ecuación. Aquí se hace paso a paso para mejor comprensión.

# Reconstruccion del termopar.
# Coeficientes directos:
# https://srdata.nist.gov/its90/type_j/jcoefficients.html
tipoJ = np.array([0.503811878150E-1,
                  0.304758369300E-4,
                 -0.856810657200E-7,
                  0.132281952950E-9,
                 -0.170529583370E-12,
                  0.209480906970E-15,
                 -0.125383953360E-18,
                  0.156317256970E-22
])
E = lambda T: tipoJ.dot([T, T**2, T**3, T**4, T**5, T**6, T**7, T**8])
# Coeficientes inversos:
invsJ = np.array([1.978425E1,
                  -2.001204E-1,
                   1.036969E-2,
                  -2.549687E-4,
                   3.585153E-6,
                  -5.344285E-8,
                   5.099890E-10
])
T90 = lambda E: invsJ.dot([E, E**2, E**3, E**4, E**5, E**6, E**7])
# Compensación de unión fria.
Vm_par = (1/500)*(senales_daq[:, 0] + 9)*1E3 - E(senales_daq[:, 1])  # en milivoltios.
T_par = T90(Vm_par)
plt.figure(11)
plt.clf()
plt.plot(T_par)

# Reconstrucción de la NTC
Vm_ntc = (1/1.3)*(senales_daq[:, 2] + 5)
R_ntc = (Vm_ntc * ntc.R_lim)/(ntc.Vcc - Vm_ntc)
T_ntc = 1./((1/ntc.B)*np.log(R_ntc/ntc.Ro)+(1/ntc.To)) - 273.15  # en Celsius.
plt.figure(12)
plt.clf()
plt.plot(T_ntc)

# Reconstrucción del RTD
Vm_rtd = (1/20)*(senales_daq[:, 3] + 10)
R_rtd = (Vm_rtd * rtd.R_lim)/(rtd.Vcc - Vm_rtd)
T_rtd = (1/(rtd.Ro * rtd.alpha))*(R_rtd - rtd.Ro)
plt.figure(13)
plt.clf()
plt.plot(T_rtd)
plt.legend(['T TERMOCUPLA', 'T NTC', 'T RTD'])
plt.title('Temperaturas Recuperadas de la Adquisición')