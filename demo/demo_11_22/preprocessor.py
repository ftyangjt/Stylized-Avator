import torch
import numpy as np
from transformers import pipeline
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

#np数组,array(H,W,C)，预处理数据，比如蒙版处理
#original_image = np.array(original_image)
#output_image = Image.fromarray(output_image)
#-----------------------------------------------------------------------
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
                #mask人脸保护区域，bigger是渐变准备区域
                small_mask = mask
                mask = cv2.erode(mask, np.ones((int(bboxC.height * iw//25), int(bboxC.width * ih//100)), np.uint8), iterations=5)
                
                
                print(f'{bboxC.width}宽度')

        


    
    #蒙版转换为PIL图像
    mask_image = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    small_mask = Image.fromarray(cv2.cvtColor(small_mask, cv2.COLOR_BGR2RGB))
    mask_image.save("./AAA/0A1.png")
    small_mask.save("./AAA/small_mask.png")


    #清理资源
    face_detection.close()
    print("人脸蒙版构建")
    return mask_image,small_mask

def blur_mask(mask_in,power=41):
    # 将输入图像转为灰度图
    mask = mask_in.convert("L")
    mask = np.array(mask)
    blurred_mask = cv2.GaussianBlur(mask, (power, power), 0) 

    norm_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)

    final_mask =norm_mask
    final_mask += cv2.bitwise_and(mask, mask, mask=mask)
    

    final_mask = Image.fromarray(final_mask)
    final_mask.save("./AAA/0A2.png")
    print("模糊蒙版")
    return final_mask

#人脸修复
def face_repairer(original_image, modified_image, mask_image):
    # 转换图像为 NumPy 数组
    original_image = np.array(original_image)
    modified_image = np.array(modified_image)
    mask_image = np.array(mask_image)

    #确保所有图像的尺寸一致
    if original_image.shape != modified_image.shape or original_image.shape[:2] != mask_image.shape[:2]:
        raise ValueError("图像和蒙版的尺寸必须匹配")

    #将图像和蒙版转换为浮点格式，并归一化到 [0, 1]
    original_image = original_image.astype(np.float32) / 255.0
    modified_image = modified_image.astype(np.float32) / 255.0
    mask_image = mask_image.astype(np.float32) / 255.0

    #蒙版是单通道灰度图，扩展为3通道
    if mask_image.ndim == 2:
        mask_image = np.stack([mask_image] * 3, axis=-1)

    #加权融合，这一步需要0-1的np数组，这实现了通过模糊蒙版的柔和过度
    #原图权重
    weigh = 0.35
    output_image = original_image * (1 - mask_image) * weigh  + modified_image * (mask_image * weigh - weigh + 1)

    # 转换回 [0, 255] 的范围，并转换为整数
    output_image = (output_image * 255).astype(np.uint8)

    # 转换为 PIL 图像
    output_image = Image.fromarray(output_image)
    output_image.save("./AAA/0A5.png")

    print("保护人脸与渐变算法完成")
    return output_image

#手保护（没有成功）
def detect_hands(image):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #创建空白蒙版
    mask = np.zeros_like(image)
    #使用 Hands 模块
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:
        #检测手
        results = hands.process(image)

        #如果检测到手
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #获取边界框
                h, w, _ = image.shape
                x1 = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y1 = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x2 = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y2 = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                #绘制矩形框
                mask[y1:y2, x1:x2] = 255

                #绘制手部关键点
                #mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            print("未检测到手。")
    mask_image = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    mask_image.save("./AAA/0A6.png")


#肖像处理蒙版，精细的人像抠图
def portrait_mask(in_image, mask_kind_list, in_mask=None):
    # 定义分割类型及颜色
    #mask_kind = 3
    CATEGORY_COLORS = {
        0: ("background", (192, 192, 192)),  # 灰
        1: ("hair", (255, 0, 0)),           # 红
        2: ("body-skin", (0, 255, 0)),      # 绿
        3: ("face-skin", (0, 0, 255)),      # 蓝
        4: ("clothes", (255, 255, 0)),      # 青
        5: ("others", (128, 0, 128))        # 紫
    }

    # 创建输出文件夹
    OUTPUT_DIR = "TEST_output_masks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建分割器选项
    #base_options = python.BaseOptions(model_asset_path='./ControlNet_models/selfie_multiclass_256x256.tflite')
    #真是奇怪，一直可以使用，但是有一次开始突然报错。GitHub上说这个在windows上有诡异问题。我没有找到问题的原因
    base_options = python.BaseOptions(model_asset_buffer=open('./ControlNet_models/selfie_multiclass_256x256.tflite', "rb").read())
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                        output_category_mask=True)

    # 创建分割器
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        
        # 加载图片
        in_image.convert("RGB")
        in_image  = np.array(in_image)

        # 指定图像格式（RGB）
        image_format = mp.ImageFormat.SRGB

        # 使用 mediapipe 创建图像对象
        image = mp.Image(image_format, in_image)
        
        # 获取分割结果
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask.numpy_view()
        
        # 读取原图像数据
        image_data = image.numpy_view()
        h, w, _ = image_data.shape
        
        # 初始化分割结果图像
        if in_mask != None:
            segmented_image = np.array(in_mask)
        else:
            segmented_image = np.ones_like(image_data, dtype=np.uint8) * 255
        
        # 遍历每种分割类型并填充
        for mask_kind in mask_kind_list:
            if mask_kind == 3:#脸
                mask = (category_mask == 3)
                segmented_image[mask] = (0,0,0)
                face_segmented_image = segmented_image.copy()
            
            if mask_kind == 5:#首饰
                mask = (category_mask == 5)
                segmented_image[mask] = (0,0,0)  
                
            if mask_kind == 4:#衣服
                mask = (category_mask == 4)
                segmented_image[mask] = (0,0,0)

            if mask_kind == 2:#身体
                mask = (category_mask == 2)
                segmented_image[mask] = (0,0,0)

        mask_image = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        mask_image.save("./AAA/segmented_image.png")
        face_segmented_image = segmented_image = Image.fromarray(cv2.cvtColor(face_segmented_image, cv2.COLOR_BGR2RGB))
        face_segmented_image = segmented_image.save("./AAA/face_and_glass_segmented_image.png")
        print(f"精细蒙版完成，蒙版类别{mask_kind_list}")
        return mask_image

def TEST():
    # 定义分割类型及颜色
    CATEGORY_COLORS = {
        0: ("background", (192, 192, 192)),  # 灰
        1: ("hair", (255, 0, 0)),           # 红
        2: ("body-skin", (0, 255, 0)),      # 绿
        3: ("face-skin", (0, 0, 255)),      # 蓝
        4: ("clothes", (255, 255, 0)),      # 青
        5: ("others", (128, 0, 128))        # 紫
    }

    # 创建输出文件夹
    OUTPUT_DIR = "TEST_output_masks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建分割器选项
    base_options = python.BaseOptions(model_asset_buffer=open('./ControlNet_models/selfie_multiclass_256x256.tflite', "rb").read())
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                        output_category_mask=True)

    # 创建分割器
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image_file_name = "./picture/stand/weixin_20241103130312.jpg"
        print(f"Processing: {image_file_name}")
        
        # 加载图片
        image = mp.Image.create_from_file(image_file_name)
        
        # 获取分割结果
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask.numpy_view()
        
        # 读取原图像数据
        image_data = image.numpy_view()
        h, w, _ = image_data.shape
        
        # 初始化分割结果图像
        segmented_image = np.zeros_like(image_data, dtype=np.uint8)
        
        # 遍历每种分割类型并填充对应的颜色
        for category_id, (label_name, color) in CATEGORY_COLORS.items():
            mask = (category_mask == category_id)
            segmented_image[mask] = color
        
        # 显示分割结果
        cv2.imshow(f"Segmented {image_file_name}", segmented_image)
        
        # 保存分割结果到磁盘
        output_path = os.path.join(OUTPUT_DIR, f"segmented_{os.path.basename(image_file_name)}")
        cv2.imwrite(output_path, segmented_image)
        print(f"Saved segmented image to {output_path}")
        
        # 分别生成每种类型的蒙版
        for category_id, (label_name, color) in CATEGORY_COLORS.items():
            mask = (category_mask == category_id).astype(np.uint8) * 255
            mask_image = np.stack([mask] * 3, axis=-1)  # 转换为三通道
            mask_output_path = os.path.join(OUTPUT_DIR, f"{label_name}_mask_{os.path.basename(image_file_name)}")
            cv2.imwrite(mask_output_path, mask_image)
            print(f"Saved {label_name} mask to {mask_output_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
#image = Image.open("picture/stand/新建文件夹/微信图片_20241118234203.jpg")
#detect_hands(image)
#TEST()