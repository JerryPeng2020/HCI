import cv2
import mediapipe as mp

# 初始化MediaPipe面部检测模块
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置MediaPipe面部检测的最小检测置信度
face_detection.min_detection_confidence = 0.7

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换颜色空间从BGR到RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 将帧数据传递给MediaPipe进行面部检测
    results = face_detection.process(frame_rgb)

    # 在检测到的人脸周围绘制矩形框
    if results.detections:
        for detection in results.detections:
            # 将MediaPipe的坐标转换为OpenCV的坐标格式
            bbox = detection.location_data.relative_bounding_box
            x, y, width, height = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])
            # 在人脸上绘制矩形框
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            # add a keypoint
            pt = detection.location_data.relative_keypoints
            print(pt[0])
            x = int(pt[0].x * frame.shape[1])
            y = int(pt[0].y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) 
    # 显示结果
    cv2.imshow('MediaPipe Face Detection', frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
face_detection.close()
cap.release()
cv2.destroyAllWindows()