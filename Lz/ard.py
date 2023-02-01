import cv2                              # библиотека opencv (получение и обработка изображения)
import mediapipe as mp                  # библиотека mediapipe (распознавание рук)
import serial                           # библиотека pyserial (отправка и прием информации)


camera = cv2.VideoCapture(0)            # получаем изображение с камеры (0 - порядковый номер камеры в системе)
video_width = 640
video_height = 480
camera.set(3, video_width) # 3 is the id for width
camera.set(4, video_height) # 4 is the id for height



mpHands = mp.solutions.hands            # подключаем раздел распознавания рук
mpHands1 = mp.solutions.hands
hands = mpHands.Hands()                 # создаем объект класса "руки"
hands1 = mpHands1.Hands()
mpDraw = mp.solutions.drawing_utils     # подключаем инструменты для рисования
mpDraw1 = mp.solutions.drawing_utils

portNo = "COM14"                         # указываем последовательный порт, к которому подключена Arduino
uart = serial.Serial(portNo, 9600)      # инициализируем последовательный порт на скорости 9600 Бод


p = [0 for i in range(21)]              # создаем массив из 21 ячейки для хранения высоты каждой точки
p1 = [0 for i in range(21)]
finger = [0 for i in range(5)]          # создаем массив из 5 ячеек для хранения положения каждого пальца
finger1 = [0 for i in range(5)]

# функция, возвращающая расстояние по модулю (без знака)
def distance(point1, point2):
    return abs(point1 - point2)


while True:
    good, img = camera.read()                                   # получаем один кадр из видеопотока
    roi = img[0 : 300, 0 : 300]
    roi1=img[0:300,400:700]
    imgRGB = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)               # преобразуем кадр в RGB
    imgRGB1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB)


    results = hands.process(imgRGB)                             # получаем результат распознавания
    if results.multi_hand_landmarks:                            # если обнаружили точки руки
        for handLms in results.multi_hand_landmarks:            # получаем координаты каждой точки

            # при помощи инструмента рисования проводим линии между точками
            mpDraw.draw_landmarks(roi, handLms, mpHands.HAND_CONNECTIONS)

            # работаем с каждой точкой по отдельности
            # создаем список от 0 до 21 с координатами точек
            for id, point in enumerate(handLms.landmark):
                # получаем размеры изображения с камеры и масштабируем
                width, height, color = roi.shape
                width, height = int(point.x * height), int(point.y * width)

                p[id] = height           # заполняем массив высотой каждой точки

            # получаем расстояние, с которым будем сравнивать каждый палец
            distanceGood = distance(p[0], p[5]) + (distance(p[0], p[5]) / 2)
            # заполняем массив 1 (палец поднят) или 0 (палец сжат)
            finger[1] = 1 if distance(p[0], p[8]) > distanceGood else 0
            finger[2] = 1 if distance(p[0], p[12]) > distanceGood else 0
            finger[3] = 1 if distance(p[0], p[16]) > distanceGood else 0
            finger[4] = 1 if distance(p[0], p[20]) > distanceGood else 0
            finger[0] = 1 if distance(p[4], p[17]) > distanceGood else 0

            # готовим сообщение для отправки
            print(finger[0],finger[1],finger[2],finger[3],finger[4])
            msg = ''
            # 0 - большой палец, 1 - указательный, 2 - средний, 3 - безымянный, 4 - мизинец
            # жест "коза" - 01001
            if not (finger[0]) and finger[1] and not (finger[2]) and not (finger[3]) and finger[4]:
                msg = '@'
            if finger[0] and not (finger[1]) and not (finger[2]) and not (finger[3]) and not (finger[4]):
                msg = '^'
            if not(finger[0]) and finger[1] and finger[2] and not(finger[3]) and not(finger[4]):
                msg = '$' + str(width) + ';'
            if not(finger[0]) and finger[1] and not(finger[2]) and not(finger[3]) and not(finger[4]):
                msg = '#' + str(width) + ';'

            # отправляем сообщение в Arduino
            if msg != '':
                msg = bytes(str(msg), 'utf-8')
                #uart.write(msg)
                print(msg)
                
                
                