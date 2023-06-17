import cv2
import os

# Función para recortar un video frame a frame y guardar los cuadros como imágenes PNG
def recortar_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Guardar el cuadro como imagen PNG en la carpeta de salida
        output_path = os.path.join(output_folder, f"frame_{frame_count}.png")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Parámetros de entrada
video_path = 'output_video.avi'
output_folder = 'fotos'

# Llamar a la función para recortar el video y guardar los cuadros como imágenes
recortar_video(video_path, output_folder)
