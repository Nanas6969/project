import cv2
import datetime
import time
import numpy as np

# Настройки
MIN_AREA = 1500           
COOLDOWN_SECONDS = 3      
LOG_FILE = "motion_log.txt"

def find_working_camera():
    # Проверяем индексы 0, 1 и 2 через range(3)
    for index in range(3):
        print(f"Проверяю камеру с индексом {index}...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1.5)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and np.mean(frame) > 1.0:
                print(f"Успех! Найдена рабочая камера: индекс {index}")
                return cap
        cap.release()
    return None

cap = find_working_camera()

if cap is None:
    print("\nОШИБКА: Рабочая камера не найдена или все камеры отдают черный экран.")
    print("Проверьте: 1) Физическую шторку на камере; 2) Клавишу отключения камеры на клавиатуре (Fn + F6/F8/F10); 3) Настройки конфиденциальности Windows.")
    exit()

first_frame = None
last_logged_time = 0

print("Скрипт запущен. Нажмите 'q' в окне видео для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Потерян сигнал с камеры.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    frame_delta = cv2.absdiff(first_frame, gray)
    # Распаковываем кортеж через _, thresh
    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    current_time = time.time()
    if motion_detected and (current_time - last_logged_time > COOLDOWN_SECONDS):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"Зафиксировано движение: {timestamp}\n"
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
            
        print(log_entry.strip())
        last_logged_time = current_time

    first_frame = gray
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
