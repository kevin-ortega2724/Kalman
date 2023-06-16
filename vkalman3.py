import cv2
import numpy as np

# Cargar la imagen de entrada
imagen = cv2.imread('IMG_1.png')

esquina = (1000, -200)
tamano = (150, 150)
roi = imagen[esquina[1]:esquina[1] + tamano[1], esquina[0]:esquina[0] + tamano[0], :]
cv2.imwrite('IMG_1.png', roi)
template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

# Inicializar el filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
kalman.statePre = np.array([esquina[0], esquina[1], 0, 0], dtype=np.float32)

frame_counter = 1

while True:
    frame_counter += 1

    # Leer el siguiente fotograma del video
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Actualizar la posición del ROI para seguir la pelota
    roi_position = (esquina[0] + top_left[0], esquina[1] + top_left[1])

    # Actualizar la medida del filtro de Kalman con la posición de la pelota
    measurement = np.array([[np.float32(roi_position[0])], [np.float32(roi_position[1])]])
    kalman.correct(measurement)

    # Predecir la siguiente posición de la pelota con el filtro de Kalman
    prediction = kalman.predict()
    predicted_position = (prediction[0], prediction[1])

    # Dibujar el rectángulo en la posición estimada por el filtro de Kalman
    cv2.rectangle(frame, (int(predicted_position[0]), int(predicted_position[1])),
                  (int(predicted_position[0]) + tamano[0], int(predicted_position[1]) + tamano[1]), (0, 255, 0), 2)
    cv2.rectangle(frame, esquina, (esquina[0] + tamano[0], esquina[1] + tamano[1]), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
