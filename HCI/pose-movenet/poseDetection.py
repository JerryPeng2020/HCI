import cv2
import tensorflow as tf
import numpy as np

# 载入MoveNet的Lightning模型
interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()

# 获取模型输入输出的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 输入数据需要的scale和zero_point
input_scale, input_zero_point = input_details[0]['quantization']

def detect_pose(frame):
    # 原始帧尺寸
    original_height, original_width, _ = frame.shape

    # 调整帧大小和颜色通道顺序，得到192x192尺寸
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.uint8)

    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

    # 执行推理
    interpreter.invoke()

    # 获取输出数据
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # 计算缩放比例和padding修正
    scale = min(192 / original_width, 192 / original_height)
    pad_x = (192 - original_width * scale) / 2
    pad_y = (192 - original_height * scale) / 2

    # 修正关键点坐标
    keypoints = []
    for keypoint in keypoints_with_scores:
        y, x, score = keypoint
        x = (x * 192 - pad_x) / scale
        y = (y * 192 - pad_y) / scale
        keypoints.append((int(x), int(y), score))

    return keypoints

# 使用OpenCV捕捉摄像头数据
cap = cv2.VideoCapture(0)
# 在while循环中绘制关键点
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 检测姿态
    keypoints = detect_pose(frame)

    # 可视化关键点
    for x, y, score in keypoints:
        if score > 0.2:  # 可以调整这个阈值来过滤不确定的关键点
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # 显示结果
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
