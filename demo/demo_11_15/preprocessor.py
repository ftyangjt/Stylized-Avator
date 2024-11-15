import torch
import numpy as np
from transformers import pipeline
import cv2
from PIL import Image
import mediapipe as mp

#np数组,array(H,W,C)，预处理数据，比如蒙版处理
#torch张量(N,C,H,W),神经网络的输入输出,通常使用0到1而非0到255
#PIL,python打开保存图片的方式，可以理解为(H,W,C)

#openCV相关默认BGR而非RGB,openCV提供颜色转换，特征识别等功能,openCV储存格式就是np


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
    #save = False
    save= True
    image = depth_estimator(init_image)["depth"]
    if save:
        # 确保输入是三通道图像，并转换为RGB格式
        image = np.array(image)
        if image.ndim == 2:  # 如果是灰度图，转为三通道
            image = np.stack([image] * 3, axis=-1)

        # 归一化并转为PyTorch Tensor
        depth_map = torch.from_numpy(image).float() / 255.0
        depth_map = depth_map.permute(2, 0, 1).unsqueeze(0).half()  # 转换为 (1, C, H, W)

        # 保存测试结果
        test_image = tensor_to_pil(depth_map.squeeze(0).cpu())  # 先去掉batch维度，再转回PIL
        test_image.save("./AAA/0C1.png")

    else:
        depth_map = image
    print("深度图产生")
    return depth_map


#脸部蒙版
def create_face_mask(pil_image, face_mask_class):
    # 初始化MediaPipe人脸检测模块
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.05)
    #1远景，0近景
    #最低置信度

    #将PIL图像转换为np
    image = np.array(pil_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #检测人脸
    results = face_detection.process(image_rgb)

    #创建空白蒙版
    mask = np.zeros_like(image)

    if results.detections:
        for detection in results.detections:
            #获取人脸边界框
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)
            
            #在蒙版上绘制black矩形表示人脸区域
            mask[y1:y2, x1:x2] = 255
            if face_mask_class == 1:
                pass
            else:
                mask = cv2.bitwise_not(mask)
                mask = cv2.erode(mask, np.ones((int(bboxC.width * iw//25), int(bboxC.height * ih//25)), np.uint8), iterations=5)
                print(f'{bboxC.width}宽度')

        


    
    #蒙版转换为PIL图像
    mask_image = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    mask_image.save("./AAA/0A1.png")


    #清理资源
    face_detection.close()
    print("人脸蒙版构建")
    return mask_image


def blur_mask(mask_in):
    # 将输入图像转为灰度图
    mask = mask_in.convert("L")
    mask = np.array(mask)

    #创建一个渐变掩码，避免硬性黑白分界
    #使用高斯模糊来生成一个渐变过渡区域

    
    #使用掩码增强黑色部分并加上原始模糊图像的白色部分
    #expanded_black_mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=5)  # 扩展黑色区域,此部分已经转移向生成蒙版程序
    blurred_mask = cv2.GaussianBlur(mask, (35, 35), 0)
    
    #归一化处理使得过渡区域更加平滑
    norm_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)

    final_mask =norm_mask
    #final_mask = cv2.bitwise_and(norm_mask, norm_mask, mask=norm_mask) # 保留扩展后的黑色部分
    final_mask += cv2.bitwise_and(mask, mask, mask=mask) # 加入模糊后的白色部分
    

    #将结果转换为PIL图像，并保存
    final_mask = Image.fromarray(final_mask)
    final_mask.save("./AAA/0A2.png")
    
    return final_mask
