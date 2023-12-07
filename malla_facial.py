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

# -------------Creamos el while principal-----------------
while True:
    ret, frame = cap.read()
    # -----Correccion de color paso de BGR a RGB
    framesRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --Obseramos los resultados---
    resultados = MallaFacial.process(framesRGB)

    # Creamos unas listas para almacenar los resultados
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:  # Si detectamos algun rostro
        for rostros in resultados.multi_face_landmarks:  # Mostramos el rostro detectado
            mpDibujo.draw_landmarks(
                frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)
            # FACE_CONNECTIONS
            # Extraer los puntos del rostro detectado
            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape  # Obtengo el ancho y alto de la ventana
                # multiplico la proporcion de la imagen por el ancho y por el alto para obtener las cordenadas del pixel
                x, y = int(puntos.x*an), int(puntos.y*al)
                px.append(x)
                px.append(y)
                lista.append([id, x, y])
                if len(lista) == 468:
                    # Ceja derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    # Ceja izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # Boca extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)

                    # Boca apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                    # Clasificacion
                    # Bravo
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Enojada', (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Feliz
                    if longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20:
                        cv2.putText(frame, 'Persona Feliz', (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    if longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 90 and longitud4 > 20:
                        cv2.putText(frame, 'Persona Asombrada', (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    if longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 95 and longitud3 > 90 and longitud4 < 5:
                        cv2.putText(frame, 'Persona triste', (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Reconocimiento de Emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
