"""
Modulo para implementar el filtro de Kalman
para el seguimiento de los parametros de localizacion

"""

import numpy as np


class Kalman:

    def __init__(self, x_ini=None, P_ini=None):
        # modelo dinamico:
        # x[k+1] = F x[k] + w
        # modelo de medida:
        # z = H x + v
        # donde:
        # w ~ N(0, Q)
        # v ~ N(0, R)

        # vector de estados
        # x = [x, y, theta]

        # Prior:
        self.xprior = np.array([0.0, 0.0, 0.0])
        self.Pprior = np.eye(3, dtype=float)
        # Posterior:
        self.x = np.array([0.0, 0.0, 0.0])
        self.P = np.eye(3, dtype=float)
        # si hay valores, usarlos:
        if x_ini is not None:
            self.x = x_ini
        if P_ini is not None:
            self.P = P_ini
        # F = identidad
        self.F = np.eye(3, dtype=float)
        # H : Matriz de medida
        self.H = np.eye(3, dtype=float)
        # Covarianzas
        # Covarianza de proceso
        self.Q = 1E-2 * np.eye(3, dtype=float)
        # Covarianza de medida
        self.R = 1E-2 * np.eye(3, dtype=float)

    def predecir(self):
        self.xprior = self.F @ self.x
        self.Pprior = self.F @ self.P @ self.F.T + self.Q

    def actualizar(self, medicion):
        # Actualizar el estado basado en la medición
        innovacion = medicion - np.dot(self.H, self.xprior)
        innovacion_covarianza = np.dot(np.dot(self.H, self.Pprior), self.H.T) + self.R
        kalman_gain = np.dot(np.dot(self.Pprior, self.H.T), np.linalg.inv(innovacion_covarianza))

        self.x = self.xprior + np.dot(kalman_gain, innovacion)
        self.P = np.dot(np.eye(len(self.x)) - np.dot(kalman_gain, self.H), self.Pprior)

        return self.x


if __name__ == "__main__":

    xf = np.array([630.0, 300.0, 0.0])
    Pf = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 5.0*np.pi/180.0]])
    filtro = Kalman(x_ini=xf, P_ini=Pf)

    filtro.predecir()

    medida = np.array([620.0, 310.0, 0.01])
    filtro.actualizar(zk=medida)

    print(filtro.x)
    print(filtro.P)