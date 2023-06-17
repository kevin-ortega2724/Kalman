import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Clase KalmanFilterParabolic para el filtro de Kalman
class KalmanFilterParabolic:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        # Definir las matrices y valores iniciales del filtro de Kalman
        self.kf.F = np.array([[1, 0, 1, 0, 0.5, 0],
                              [0, 1, 0, 1, 0, 0.5],
                              [0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 1, 0, 1],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0]])

        self.kf.P *= 1000
        self.kf.R = np.array([[10, 0],
                              [0, 10]])

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)

    def get_estimated_state(self):
        return self.kf.x[:4]

# Función para detectar objetos azules en un cuadro de imagen
def detect_blue_objects(frame):
    # Convertir el cuadro de imagen de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir el rango de color azul en HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Crear una máscara para los píxeles azules
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicar operaciones morfológicas para mejorar la detección
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos de los objetos azules
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar una lista para almacenar los rectángulos de los objetos detectados
    rectangles = []

    for contour in contours:
        # Obtener el rectángulo del contorno
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrar los rectángulos pequeños (posibles falsos positivos)
        if w > 20 and h > 20:
            rectangles.append((x, y, w, h))

    return rectangles
# Cargar el video
video_path = 'VID.mp4'
cap = cv2.VideoCapture(video_path)

# Obtener las propiedades del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear el objeto para guardar el video con el movimiento
output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Crear una instancia de la clase KalmanFilterParabolic
kf_parabolic = KalmanFilterParabolic()

# Lista para almacenar las estimaciones de posición y velocidad
estimations = []

# Procesar cada cuadro del video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detectar objetos azules en el cuadro
    rectangles = detect_blue_objects(frame)

    # Obtener las coordenadas (posición x, posición y) de los rectángulos de los objetos detectados
    measurements = []
    for (x, y, w, h) in rectangles:
        center_x = x + w/2
        center_y = y + h/2
        measurements.append(np.array([center_x, center_y], dtype=float))

    # Procesar cada medición con el filtro de Kalman
    for measurement in measurements:
        # Realizar la predicción del filtro de Kalman
        kf_parabolic.predict()

        # Actualizar el filtro de Kalman con la medición
        kf_parabolic.update(measurement)

        # Obtener el estado estimado del filtro de Kalman
        estimated_state = kf_parabolic.get_estimated_state()
        estimated_position = estimated_state[:2]
        estimated_velocity = estimated_state[2:]

        # Guardar las estimaciones de posición y velocidad
        estimations.append((estimated_position[0], estimated_position[1],
                            estimated_velocity[0], estimated_velocity[1]))

        # Dibujar el estado estimado en el cuadro procesado
        estimated_x, estimated_y = estimated_position
        cv2.circle(frame, (int(estimated_x), int(estimated_y)), 5, (0, 0, 255), -1)
        
        # Dibujar el recuadro de seguimiento
        x, y, w, h = rectangles[0]  # Se asume que solo se detecta un objeto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  

    # Guardar el cuadro procesado en el video de salida
    out.write(frame)

    # Mostrar el cuadro procesado en una ventana
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar las estimaciones en un archivo CSV
csv_path = 'estimations.csv'
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['TSTAMP', 'EstX', 'EstY', 'EstVX', 'EstVY'])
    for i, estimation in enumerate(estimations):
        tstamp = i / fps  # Timestamp basado en el número de cuadro y la frecuencia de cuadros por segundo
        writer.writerow([tstamp, *estimation])

# Graficar el movimiento estimado
timestamps = [i / fps for i in range(len(estimations))]
positions_x = [estimation[0] for estimation in estimations]
positions_y = [estimation[1] for estimation in estimations]
velocities_x = [estimation[2] for estimation in estimations]
velocities_y = [estimation[3] for estimation in estimations]

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(timestamps, positions_x, label='EstX')
plt.plot(timestamps, positions_y, label='EstY')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()

plt.subplot(212)
plt.plot(timestamps, velocities_x, label='EstVX')
plt.plot(timestamps, velocities_y, label='EstVY')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
