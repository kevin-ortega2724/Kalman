"""

"""
import cv2

cap = cv2.VideoCapture('VID.mp4')
esquina = (630, 300)
tamano = (150, 150)
ret, frame = cap.read()
roi = frame[esquina[1]:esquina[1] + tamano[1], esquina[0]:esquina[0] + tamano[0], :]
cv2.imwrite('IMG.jpg', roi)
template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

frame_counter = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_counter += 1

    # If the last frame is reached:
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # All the 6 methods for Template Matching in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(gray, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()