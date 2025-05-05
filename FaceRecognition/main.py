import face_recognition
import cv2

# Cargar las imágenes de referencia y sus codificaciones
imagenes_referencia = ["ejemplo.jpg", "ejemplo1.jpg", "ejemplo2.jpg","benny.png"]
encodings_referencia = []
nombres_referencia = ["jos5", "brattpit", "Persona 3","benceurs"]  # Nombres asociados a cada imagen

# Cargar y codificar las imágenes de referencia
for imagen in imagenes_referencia:
    foto_referencia = face_recognition.load_image_file(imagen)
    encoding_referencia = face_recognition.face_encodings(foto_referencia)
    if encoding_referencia:  # Asegurarse de que la imagen contiene una cara
        encodings_referencia.append(encoding_referencia[0])

# Usar la webcam
cap = cv2.VideoCapture(1)

# Configurar la resolución de la cámara (reduce la carga de procesamiento)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Presiona 'c' para capturar y comparar la cara")

while True:
    ret, frame = cap.read()

    # Reducir la resolución de la imagen de la cámara para acelerar el procesamiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detectar las caras en el cuadro reducido
    face_locations = face_recognition.face_locations(small_frame)

    # Solo procesar las caras detectadas
    if face_locations:
        # Extraer las características faciales de las caras detectadas
        encodings_live = face_recognition.face_encodings(small_frame, face_locations)

        # Comparar las características faciales con las imágenes de referencia
        for encoding_live, face_location in zip(encodings_live, face_locations):
            coincidencias = face_recognition.compare_faces(encodings_referencia, encoding_live)
            
            # Dibujar un cuadro alrededor de la cara detectada
            top, right, bottom, left = [int(val * 4) for val in face_location]  # Escalar las coordenadas
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Si alguna cara coincide, mostrar el nombre correspondiente
            if any(coincidencias):  
                match_index = coincidencias.index(True)
                cv2.putText(frame, f"Coincide con: {nombres_referencia[match_index]}", 
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "No coincide con ninguna persona", 
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el video en vivo
    cv2.imshow('Captura en vivo', frame)

    # Esperar a que el usuario presione 'c' para capturar y comparar
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print("Comparando...")
        break

cap.release()
cv2.destroyAllWindows()
