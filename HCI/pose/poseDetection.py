import cv2
import mediapipe as mp
import numpy as np  # 导入 NumPy 库

# 初始化MediaPipe姿态检测模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    enable_segmentation=True  # 启用分割掩码
)

# 初始化绘图模块
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换颜色从BGR到RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像检测姿态
    results = pose.process(frame_rgb)

    # 将结果绘制回图像上
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            )
        )

    # 绘制分割掩码
    if 'segmentation_mask' in dir(results) and results.segmentation_mask is not None:
        segmentation_mask = results.segmentation_mask
        # 将布尔值分割掩码转换为0和255的单通道图像
        segmentation_mask_image = (segmentation_mask * 255).astype(np.uint8)
        # 创建一个三通道的绿色掩码图像
        green_mask = np.zeros_like(frame_rgb)  # 创建一个和frame_rgb相同大小的黑色图像
        green_mask[:, :, 1] = segmentation_mask_image  # 将分割掩码设置为绿色
        # 应用透明度
        frame_with_mask = cv2.addWeighted(frame, 0.5, green_mask, 0.8, 0)

    # 显示图像
    cv2.imshow('Pose Detection with Segmentation Mask', frame_with_mask)

    # 按下 'Esc' 键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
pose.close()
cap.release()
cv2.destroyAllWindows()