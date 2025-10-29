import cv2
import subprocess as sp
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FACE_CASCADE = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")

WIDTH, HEIGHT = 640, 480
FPS = 30
CAMERA_DEVICE = 0

# Ajuste conforme sua rede
RTP_OUT_URL = 'udp://0.0.0.0:5000' # udp://(IP do RECEPTOR):(PORTA)

FFMPEG_CMD = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{WIDTH}x{HEIGHT}',
    '-r', str(FPS),
    '-i', '-',
    '-c:v', 'mpeg4',
    '-q:v', '5',
    '-pix_fmt', 'yuv420p',
    '-f', 'mpegts',
    RTP_OUT_URL
]

face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
if face_cascade.empty():
    print(f"ERRO CRÍTICO: Não foi possível carregar o classificador: {FACE_CASCADE}", file=sys.stderr)
    sys.exit(1)

cap = cv2.VideoCapture(CAMERA_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("ERRO CRÍTICO: Não foi possível abrir o dispositivo de câmera V4L2.", file=sys.stderr)
    sys.exit(1)

print(f"INFO: Iniciando stream em: {RTP_OUT_URL}")
try:
    process = sp.Popen(FFMPEG_CMD, stdin=sp.PIPE)
except FileNotFoundError:
    print("ERRO: ffmpeg não encontrado. Instale com: sudo apt install ffmpeg", file=sys.stderr)
    sys.exit(1)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("AVISO: Falha ao ler frame da câmera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detectada', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        process.stdin.write(frame.tobytes())

except KeyboardInterrupt:
    print("\nINFO: Encerrado pelo usuário.")

finally:
    print("INFO: Liberando recursos...")
    cap.release()
    if process.poll() is None:
        process.stdin.close()
        process.terminate()
        process.wait()
    print("INFO: Encerrado.")

