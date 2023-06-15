import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
def kalman_filter(z, x0, P0, A, B, H, Q, R):
    """
    Filtro de Kalman para un sistema lineal
    
    z : array_like
        Vector de mediciones (shape: NxM)
    x0 : array_like
        Vector de estado inicial (shape: Kx1)
    P0 : array_like
        Matriz de covarianza del error inicial (shape: KxK)
    A : array_like
        Matriz de transición de estado (shape: KxK)
    B : array_like
        Matriz de entrada de control (shape: KxL)
    H : array_like
        Matriz de observación (shape: MxK)
    Q : array_like
        Matriz de covarianza del ruido del proceso (shape: KxK)
    R : array_like
        Matriz de covarianza del ruido de medición (shape: MxM)
    
    Returns
    -------
    x : array_like
        Vector de estado estimado (shape: NxK)
    """
    N = z.shape[0]  # Número de mediciones
    K = x0.shape[0]  # Número de variables de estado
    
    # Inicializar el vector de estado y la matriz de covarianza del error
    x = np.zeros((N, K))
    P = np.zeros((N, K, K))
    
    # Estimación inicial del estado y la covarianza del error
    x[0] = x0.ravel()
    P[0] = P0
    
    for k in range(1, N):
        # Predicción del estado siguiente y la covarianza del error
        x[k] = A @ x[k-1]
        P[k] = A @ P[k-1] @ A.T + Q
        
        # Cálculo del residuo y la matriz de ganancia de Kalman
        y = z[k] - H @ x[k]
        S = H @ P[k] @ H.T + R
        K = P[k] @ H.T @ inv(S)
        
        # Corrección de la estimación del estado y la covarianza del error
        x[k] += K @ y
        P[k] -= K @ H @ P[k]
    
    return x

# Parámetros del movimiento parabólico
v0 = 10  # Velocidad inicial
theta = np.pi/4  # Ángulo de lanzamiento en radianes
g = 9.8  # Aceleración debido a la gravedad

# Obtener las coordenadas x, y del movimiento parabólico real
t_flight = 2 * v0 * np.sin(theta) / g
t = np.linspace(0, t_flight, num=100)
x_real = v0 * np.cos(theta) * t
y_real = v0 * np.sin(theta) * t - 0.5 * g * t**2

# Simular mediciones ruidosas de las coordenadas x, y
sigma_z = 1  # Desviación estándar del ruido de medición
z = np.column_stack((x_real, y_real)) + sigma_z * np.random.randn(*x_real.shape).reshape(-1,1)

# Parámetros del filtro de Kalman
delta_t = t[1] - t[0]  # Intervalo de tiempo entre mediciones

A = np.array([[1, delta_t, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, delta_t],
              [0, 0, -g*delta_t**2/2 , 1]])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

Q = np.diag([1e-3]*4)   # Matriz de covarianza del ruido del proceso 
R = sigma_z**2 * np.eye(2)   # Matriz de covarianza del ruido de medición 

# Estimación inicial del estado y la covarianza del error 
v_00 = 0
x_00= z[0][0]
y_00= z[0][1]
vx_00= v_00*np.cos(theta)
vy_00= v_00*np.sin(theta)

x_00= z[0][0]
y_00= z[0][1]
vx_00= v_00*np.cos(theta)
vy_00= v_00*np.sin(theta)

x_00= z[0][0]
y_00= z[0][1]
vx_00= v_00*np.cos(theta)
vy_00= v_00*np.sin(theta)

x_00= z[0][0]
y_00= z[0][1]
vx_00= v_00*np.cos(theta)
vy_00= v_00*np.sin(theta)

x_00= z[0][0]
y_00= z[0][1]
vx_00= v_00*np.cos(theta)
vy_00= v_00*np.sin(theta)

x_init=np.array([x_00,vx_00,y_00,vx_00])
P_init=np.diag([sigma_z**2]*4)

# Aplicar el filtro de Kalman a las mediciones ruidosas 
x_estimated = kalman_filter(z,x_init,P_init,A,None,H,Q,R)

# Graficar el movimiento parabólico real y estimado 
plt.plot(x_real,y_real,label='Real')
plt.plot(z[:, 0],z[:,1],'k.',label='Mediciones')
plt.plot(x_estimated[:, 0], x_estimated[:,2],label='Estimado')
plt.title("Movimiento Parabólico Simple")
plt.xlabel("Distancia horizontal (m)")
plt.ylabel("Altura vertical (m)")
plt.grid(True)
plt.legend()
plt.show()

