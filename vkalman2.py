import cv2

cap = cv2.VideoCapture('VID.mp4')
esquina = (1000, -200)
tamano = (150, 150)
ret, frame = cap.read()
roi = frame[esquina[1]:esquina[1] + tamano[1], esquina[0]:esquina[0] + tamano[0], :]
cv2.imwrite('IMG_1.png', roi)
template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

frame_counter = 1
roi_position = esquina

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Actualizar la posición del ROI para seguir la pelota
    roi_position = (esquina[0] + top_left[0], esquina[1] + top_left[1])

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.rectangle(frame, esquina, (esquina[0] + tamano[0], esquina[1] + tamano[1]), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
