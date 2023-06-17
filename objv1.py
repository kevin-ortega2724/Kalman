import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from fk1 import KalmanFilterParabolic

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

        # Utiliza estimated_position y estimated_velocity para realizar cualquier acción deseada
        # (por ejemplo, guardar las posiciones y velocidades estimadas)
        # ...

        # Dibujar el estado estimado en el cuadro procesado
        estimated_x, estimated_y = estimated_position
        cv2.circle(frame, (int(estimated_x), int(estimated_y)), 5, (0, 0, 255), -1)

    # Guardar el cuadro procesado en el video de salida
    out.write(frame)

    # Mostrar el cuadro procesado en una ventana
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()