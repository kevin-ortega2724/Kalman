import cv2
from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanFilterParabolic:
    def __init__(self):
        # Crear el filtro de Kalman
        self.kf = KalmanFilter(dim_x=6, dim_z=2)  # 6 estados (posición x, posición y, velocidad x, velocidad y, aceleración x, aceleración y), 2 mediciones (posición x, posición y)

        # Definir las matrices del filtro de Kalman
        self.kf.x = np.array([0, 0, 0, 0, 0, 0], dtype=float)  # Estado inicial (posición x, posición y, velocidad x, velocidad y, aceleración x, aceleración y)
        self.kf.F = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=float)  # Matriz de transición: describe cómo evoluciona el estado del sistema
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=float)  # Matriz de medición: solo medimos la posición x y posición y
        self.kf.P *= 1000  # Covarianza inicial del error del estado
        self.kf.R = np.array([[0.1, 0], [0, 0.1]], dtype=float)  # Covarianza del ruido de medición
        self.kf.Q = np.array([[0.00001, 0, 0, 0, 0, 0], [0, 0.00001, 0, 0, 0, 0], [0, 0, 0.001, 0, 0, 0], [0, 0, 0, 0.001, 0, 0], [0, 0, 0, 0, 0.001, 0], [0, 0, 0, 0, 0, 0.001]], dtype=float)  # Covarianza del ruido del proceso

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)

    def get_estimated_state(self):
        return self.kf.x[:2]
