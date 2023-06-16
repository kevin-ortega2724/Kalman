import cv2
import numpy as np

# Cargar la imagen de referencia de la pelota
reference_image = cv2.imread('IMG_1A.jpg')

# Convertir la imagen de referencia a escala de grises
gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Crear el objeto del filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

# Inicializar las variables del filtro de Kalman
last_measurement = np.array([[0], [0]], np.float32)
last_prediction = np.array([[0], [0]], np.float32)

# Cargar el video
video = cv2.VideoCapture('VID.mp4')

# Obtener las dimensiones del video
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear una imagen negra con las dimensiones del video para dibujar la trayectoria estimada
trajectory_image = np.zeros((height, width, 3), dtype=np.uint8)

while True:
    # Leer el siguiente cuadro del video
    ret, frame = video.read()
    if not ret:
        break

    # Convertir el cuadro a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realizar la detección de características utilizando coincidencia de plantilla
    res = cv2.matchTemplate(gray_frame, gray_reference_image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + gray_reference_image.shape[1], top_left[1] + gray_reference_image.shape[0])

    # Obtener la posición estimada de la pelota
    x = top_left[0] + gray_reference_image.shape[1] / 2
    y = top_left[1] + gray_reference_image.shape[0] / 2
    # Dibujar el cuadro de seguimiento alrededor de la pelota
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Realizar la medición del estado
    measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(measurement)

    # Realizar la predicción del estado
    prediction = kalman.predict()

    # Obtener la posición y velocidad estimadas
    x_estimated, y_estimated = prediction[0], prediction[1]
    vx_estimated, vy_estimated = prediction[2], prediction[3]

    # Dibujar la posición y velocidad estimadas en el cuadro del video
    cv2.circle(frame, (int(x_estimated), int(y_estimated)), 10, (0, 0, 255), -1)
    cv2.putText(frame, f'Position: ({int(x_estimated)}, {int(y_estimated)})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Velocity: ({int(vx_estimated)}, {int(vy_estimated)})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Dibujar la trayectoria estimada en la imagen de referencia
    cv2.circle(trajectory_image, (int(x_estimated), int(y_estimated)), 1, (0, 0, 255), -1)

    # Mostrar los cuadros del video y la imagen de referencia
    cv2.imshow('Video', frame)
    cv2.imshow('Trajectory', trajectory_image)

    # Detener la ejecución al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video.release()
cv2.destroyAllWindows()
