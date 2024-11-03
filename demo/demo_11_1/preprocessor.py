import torch
import numpy as np
from transformers import pipeline
import cv2
from PIL import Image
import mediapipe as mp

#(1, 3, H, W)torch张量转换为PIL,保存预处理图片
def tensor_to_pil(depth_map):
    # 去掉 batch 维度并转换为形状 (H, W, 3)
    depth_map_np = depth_map.squeeze().permute(1, 2, 0).cpu().numpy()
    # 将浮点像素值（0-1）转换为 0-255 范围的 uint8 类型
    depth_map_np = (depth_map_np * 255).astype(np.uint8)
    # 转换为PIL
    pil_image = Image.fromarray(depth_map_np)
    return pil_image

#边缘图
def get_array_map(init_image):
    image_array = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2GRAY)
    image_array = np.array(image_array)
    image_array = cv2.Canny(
      image_array,
      100, #canny_lower_threshold,
      200, #canny_higher_threshold
    )
    image_array = image_array[:, :, None]
    image_array = np.concatenate([image_array, image_array, image_array], axis=2)
    image_array = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    image_array.save("./AAA/0B1.png")
    return image_array

#深度图
def get_depth_map(init_image):
    depth_estimator_model = "./ControlNet_models/models--Intel--dpt-beit-large-512"
    depth_estimator = pipeline("depth-estimation", model=depth_estimator_model, cache_dir="./cache/")

    image = depth_estimator(init_image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    depth_map = depth_map.unsqueeze(0).half()
    test_image = tensor_to_pil(depth_map)
    test_image.save("./AAA/0C1.png")

    return depth_map


#脸部蒙版
def create_face_mask(pil_image, face_mask_class):
    # 初始化MediaPipe人脸检测模块
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.15)
    #1远景，0近景
    #最低置信度

    # 将PIL图像转换为OpenCV格式
    image = np.array(pil_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 检测人脸
    results = face_detection.process(image_rgb)

    # 创建空白蒙版
    mask = np.zeros_like(image)

    if results.detections:
        for detection in results.detections:
            # 获取人脸边界框
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)
            
            # 在蒙版上绘制白色矩形表示人脸区域
            mask[y1:y2, x1:x2] = 255

    if face_mask_class == 0:
        mask = cv2.bitwise_not(mask)
    # 将蒙版转换为PIL图像
    mask_image = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    mask_image.save("./AAA/0A1.png")


    # 清理资源
    face_detection.close()

    return mask_image


def blur_mask(mask_in):
    mask = mask_in.convert("L")
    mask = np.array(mask)

    # 创建掩码：将黑色部分设为 1，其余部分设为 0
    black_mask = (mask == 0).astype(np.uint8)

    # 高斯模糊处理整个图像
    blurred_image = cv2.GaussianBlur(mask, (65, 65), 0)

    # 使用掩码将黑色部分覆盖回模糊后的图像
    final_mask = cv2.bitwise_and(mask, mask, mask=black_mask) + cv2.bitwise_and(blurred_image, blurred_image, mask=1 - black_mask)

    final_mask = Image.fromarray(cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
    final_mask.save("./AAA/0A2.png")
    return final_mask
