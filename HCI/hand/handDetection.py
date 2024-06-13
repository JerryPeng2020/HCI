import cv2
import mediapipe as mp

# 初始化MediaPipe手势检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 初始化绘图模块
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 转换颜色从BGR到RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像检测手势
    results = hands.process(frame_rgb)

    

    # 将结果绘制回图像上
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw middle finger tip
            pt = hand_landmarks.landmark[12]
            print(pt)
            x = int(pt.x * frame.shape[1])
            y = int(pt.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) 

    # 显示图像
    cv2.imshow('Hand Gesture Recognition', frame)

    # 按下 'Esc' 键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
hands.close()
cap.release()
cv2.destroyAllWindows()
