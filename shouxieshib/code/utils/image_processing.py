import cv2
import numpy as np
import torch
from typing import List, Tuple
from torchvision import transforms

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """图像预处理：去噪和对比度增强"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 调整图像大小，保持宽高比
    max_dimension = 800
    height, width = gray.shape
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height))
    
    # 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 高斯去噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Otsu's 二值化
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def segment_digits(image: np.ndarray) -> List[np.ndarray]:
    """分割图像中的数字"""
    # 复制原始图像
    image_copy = image.copy()
    
    # 寻找轮廓前先进行形态学操作
    kernel = np.ones((3,3), np.uint8)
    image_copy = cv2.morphologyEx(image_copy, cv2.MORPH_CLOSE, kernel)
    image_copy = cv2.morphologyEx(image_copy, cv2.MORPH_OPEN, kernel)
    
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(
        image_copy, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 过滤和处理轮廓
    digit_images = []
    min_contour_area = 50  # 降低最小面积阈值
    
    # 获取所有有效轮廓的边界框
    valid_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(contour)
        
        # 更宽松的过滤条件
        if area > min_contour_area and 0.1 < aspect_ratio < 3:
            valid_boxes.append((x, y, w, h))
    
    # 按x坐标排序
    valid_boxes.sort(key=lambda box: box[0])
    
    # 处理每个有效的边界框
    for x, y, w, h in valid_boxes:
        # 扩大边界框以包含完整数字
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        # 提取数字
        digit = image[y_start:y_end, x_start:x_end]
        
        # 确保提取的区域不为空
        if digit.size > 0:
            # 添加padding确保数字居中
            padding = 4
            digit_padded = np.pad(digit, padding, mode='constant', constant_values=0)
            
            # 调整为固定大小 (28x28)
            digit_resized = cv2.resize(digit_padded, (28, 28), interpolation=cv2.INTER_AREA)
            
            # 确保数字是黑底白字（与MNIST数据集一致）
            if np.mean(digit_resized[0, :]) > 127:  # 如果边缘是白色，说明需要反转
                digit_resized = 255 - digit_resized
            
            # 二值化处理
            _, digit_resized = cv2.threshold(digit_resized, 127, 255, cv2.THRESH_BINARY)
            
            digit_images.append(digit_resized)
    
    return digit_images

def prepare_for_model(image: np.ndarray) -> torch.Tensor:
    """将图像转换为模型输入格式"""
    # 确保图像是 float32 类型并归一化
    image = image.astype(np.float32) / 255.0
    
    # 添加批次和通道维度
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    # 添加与训练数据相同的标准化
    transform = transforms.Normalize((0.1307,), (0.3081,))
    tensor = transform(tensor)
    
    return tensor