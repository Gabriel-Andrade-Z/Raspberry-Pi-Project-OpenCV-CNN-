import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite("frame.jpg", frame)
    print("imagem capturada")

cap.release()
cv2.destroyAllWindows()

