import cv2
import mediapipe as mp
import math

# ------------Realizamos la VideoCaptura------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Definimos el ancho de la ventana
cap.set(4, 720)  # Definimos el alto de la ventana

# ------------Creamos nuestra funcion de dibujo-----------
mpDibujo = mp.solutions.drawing_utils
# Ajustamos la configuracion de dibujo
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# --Creamos un objeto done almacenaremos la malla facial--
mpMallaFacial = mp.solutions.face_mesh  # Primero llamamos a la funcion
# Creamos el objeto (Ctrl + Click)
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

longitud = []
# -------------Creamos el while principal-----------------
while True:
    ret, frame = cap.read()
    # -----Correccion de color paso de BGR a RGB
    framesRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --Obseramos los resultados---
    resultados = MallaFacial.process(framesRGB)

    # Creamos unas listas para almacenar los resultados
    lista = []
    r = 5
    t = 3
    if resultados.multi_face_landmarks:  # Si detectamos algun rostro
        for rostro in resultados.multi_face_landmarks:  # Mostramos el rostro detectado
            mpDibujo.draw_landmarks(
                frame, rostro, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)
            # Extraer los puntos del rostro detectado
            for id, puntos in enumerate(rostro.landmark):
                al, an, c = frame.shape  # Obtengo el ancho y alto de la ventana
                # multiplico la proporcion de la imagen por el ancho y por el alto para obtener las cordenadas del pixel
                x, y = int(puntos.x*an), int(puntos.y*al)
                lista.append([id, x, y])
            long = []
            for key1, punto1 in enumerate(lista):
                for key2, punto2 in enumerate(lista):
                    # guarda en el array de logitud la longitud entra todos los puntos en cada frame
                    long.append(math.hypot(
                        punto2[1] - punto1[1], punto2[2] - punto1[2]))
    longitud.append(long)
    cv2.imshow("Reconocimiento de Emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()

for i in longitud:
    print(len(i))
    print(i)
