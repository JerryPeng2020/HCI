import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Mesh 模块
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 初始化绘图模块
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 转换颜色从 BGR 到 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像进行人脸 mesh 检测
    results = face_mesh.process(frame_rgb)
    # print(results.multi_face_landmarks)

    # 将结果绘制回图像上
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            
            # OpenCV 绘制
            pt = face_landmarks.landmark[9]
            # print(pt.x)
            x = int(pt.x * frame.shape[1])
            y = int(pt.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) 

    # 显示图像
    cv2.imshow('Face Mesh Detection', frame)

    # 按下 'Esc' 键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
