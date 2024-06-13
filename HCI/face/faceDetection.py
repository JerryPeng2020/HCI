# pip install mediapipe opencv-python


import cv2
import mediapipe as mp

# 初始化MediaPipe人脸检测模块
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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

    # 处理图像进行人脸检测
    results = face_detection.process(frame_rgb)
    # print(results.detections)

    # 将结果绘制回图像上
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # 显示图像
    cv2.imshow('Face Detection', frame)

    # 按下 'Esc' 键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
face_detection.close()
cap.release()
cv2.destroyAllWindows()
