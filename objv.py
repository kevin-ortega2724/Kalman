import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

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
output_path = 'frames/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# Procesar cada cuadro del video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detectar objetos azules en el cuadro
    rectangles = detect_blue_objects(frame)

    # Dibujar rectángulos alrededor de los objetos detectados
    for (x, y, w, h) in rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
