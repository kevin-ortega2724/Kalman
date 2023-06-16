import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

def measure_position(x, y, sigma):
    """
    Simula una medición de posición con ruido gaussiano.
    
    x : float
        Posición en el eje x.
    y : float
        Posición en el eje y.
    sigma : float
        Desviación estándar del ruido de medición.
    
    Returns
    -------
    float
        Medición de posición con ruido.
    """
    return np.random.normal(x, sigma), np.random.normal(y, sigma)

def measure_velocity(vx, vy, sigma):
    """
    Simula una medición de velocidad con ruido gaussiano.
    
    vx : float
        Velocidad en el eje x.
    vy : float
        Velocidad en el eje y.
    sigma : float
        Desviación estándar del ruido de medición.
    
    Returns
    -------
    float
        Medición de velocidad con ruido.
    """
    return np.random.normal(vx, sigma), np.random.normal(vy, sigma)

def simulate_sensors(x_real, y_real, vx_real, vy_real, sigma_position, sigma_velocity):
    """
    Simula las mediciones de posición y velocidad con ruido gaussiano.
    
    x_real : array_like
        Vector de posiciones reales (shape: N)
    y_real : array_like
        Vector de posiciones reales (shape: N)
    vx_real : float
        Velocidad en el eje x real
    vy_real : array_like
        Vector de velocidades en el eje y reales (shape: N)
    sigma_position : float
        Desviación estándar del ruido de medición de posición.
    sigma_velocity : float
        Desviación estándar del ruido de medición de velocidad.
    
    Returns
    -------
    z : array_like
        Matriz de mediciones (shape: Nx4) donde las columnas corresponden a x_medido, y_medido, vx_medido, vy_medido
    """
    N = len(x_real)
    z = np.zeros((N, 4))
    for i in range(N):
        z[i, 0], z[i, 1] = measure_position(x_real[i], y_real[i], sigma_position)
        z[i, 2], z[i, 3] = measure_velocity(vx_real, vy_real[i], sigma_velocity)
    return z

# Parámetros del movimiento parabólico
v0 = 10  # Velocidad inicial
theta = np.pi/4  # Ángulo de lanzamiento en radianes
g = 9.8  # Aceleración debido a la gravedad

# Obtener las coordenadas x, y y las velocidades vx, vy del movimiento parabólico real
t_flight = 2 * v0 * np.sin(theta) / g
t = np.linspace(0, t_flight, num=100)
x_real = v0 * np.cos(theta) * t
y_real = v0 * np.sin(theta) * t - 0.5 * g * t**2
vx_real = v0 * np.cos(theta)
vy_real = v0 * np.sin(theta) - g * t

# Parámetros de simulación de los sensores
sigma_position = 1  # Desviación estándar del ruido de medición de posición
sigma_velocity = 0.5  # Desviación estándar del ruido de medición de velocidad

# Simular mediciones de posición y velocidad con ruido
z = simulate_sensors(x_real, y_real, vx_real, vy_real, sigma_position, sigma_velocity)

# Graficar las mediciones simuladas
plt.plot(x_real, y_real, label='Real')
plt.plot(z[:, 0], z[:, 1], 'k.', label='Mediciones')
plt.title("Simulación de Sensores para Movimiento Parabólico")
plt.xlabel("Distancia horizontal (m)")
plt.ylabel("Altura vertical (m)")
plt.grid(True)
plt.legend()
plt.show()
