# Usa una imagen de Python base
FROM python:3.10-slim

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dlib y OpenCV
RUN pip install --upgrade pip
RUN pip install dlib opencv-python

# Exponer el puerto para la cámara si es necesario (en este caso no es necesario en Windows)
EXPOSE 8080

# Comando por defecto para probar el acceso a la cámara
CMD ["python", "-c", "import cv2; print('Iniciando cámara...'); cap = cv2.VideoCapture(0); ret, frame = cap.read(); cap.release(); print('Captura exitosa!')"]
