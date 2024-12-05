import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import base64
from models.cnn_model import DigitRecognitionCNN
from utils.image_processing import preprocess_image, segment_digits, prepare_for_model

def load_model():
    model = DigitRecognitionCNN()
    model.load_state_dict(torch.load('weights/best_model.pth'))
    model.eval()
    return model

def set_page_config():
    """设置页面配置"""
    st.set_page_config(
        page_title="AI手写数字识别系统",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """添加自定义CSS样式"""
    st.markdown("""
        <style>
        /* 全局样式 */
        .main {
            padding: 2rem;
            background-color: #fafafa;
        }
        
        /* 标题样式 */
        .title-container {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .title-text {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            margin: 0;
        }
        
        .subtitle-text {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            margin-top: 1rem;
        }
        
        /* 结果卡片样式 */
        .result-box {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin: 0.5rem;
            transition: transform 0.2s;
        }
        
        .result-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* 置信度样式 */
        .confidence-high {
            color: #10B981;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .confidence-low {
            color: #EF4444;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        /* 进度条样式 */
        .stProgress > div > div {
            background-color: #10B981;
        }
        
        /* 上传区域样式 */
        .upload-container {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            border: 2px dashed #E5E7EB;
            text-align: center;
            margin: 2rem 0;
        }
        
        /* 图片显示区域 */
        .image-display {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin: 1rem 0;
        }
        
        /* 状态文本 */
        .status-text {
            color: #6B7280;
            font-size: 1rem;
            text-align: center;
            padding: 0.5rem;
        }
        
        /* 侧边栏样式 */
        .css-1d391kg {
            background-color: #F3F4F6;
        }
        
        /* 滑块样式 */
        .stSlider > div > div > div {
            background-color: #10B981;
        }
        
        /* 错误提示样式 */
        .stAlert {
            background-color: #FEE2E2;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #EF4444;
        }
        </style>
    """, unsafe_allow_html=True)

def post_process_prediction(outputs: torch.Tensor, confidence_threshold: float = 0.5):
    """对模型输出进行后处理"""
    probabilities = torch.softmax(outputs, dim=1)
    confidence, prediction = probabilities.max(1)
    
    # 如果置信度太低，尝试其他预测
    if confidence.item() < confidence_threshold:
        # 获取前三个最可能的预测
        top3_confidence, top3_predictions = torch.topk(probabilities, 3, dim=1)
        
        # 如果前三个预测的置信度差距不大，选择更可能的数字
        if (top3_confidence[0, 0] - top3_confidence[0, 1]) < 0.1:
            # 根据一些启发式规则选择更可能的数字
            # 例如：偏好常见数字，避免容易混淆的数字
            common_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            confused_pairs = {(1, 7), (3, 8), (5, 6), (2, 7)}
            
            for conf, pred in zip(top3_confidence[0], top3_predictions[0]):
                pred_num = pred.item()
                if pred_num in common_digits:
                    prediction = pred
                    confidence = conf
                    break
    
    return confidence, prediction

def main():
    set_page_config()
    add_custom_css()
    
    # 标题区域
    st.markdown("""
        <div class='title-container'>
            <h1 class='title-text'>🤖 AI手写数字识别系统</h1>
            <p class='subtitle-text'>基于深度学习的智能数字识别系统 | 准确快速</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 🎮 控制面板")
        
        uploaded_file = st.file_uploader(
            "📤 上传图片", 
            type=['png', 'jpg', 'jpeg'],
            help="支持PNG、JPG、JPEG格式的图片"
        )
        
        confidence_threshold = st.slider(
            "🎯 置信度阈值", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="调整识别结果的可信度阈值"
        )
        
        st.markdown("""
            ---
            ### 📖 使用指南
            
            1️⃣ **上传图片**
            - 选择包含手写数字的图片文件
            - 支持PNG、JPG、JPEG格式
            
            2️⃣ **调整阈值**
            - 使用滑块调整置信度阈值
            - 阈值越高，结果越可靠
            
            3️⃣ **查看结果**
            - 系统将自动识别图片中的数字
            - 显示每个数字的识别结果和置信度
            
            ---
            ### 💡 最佳实践
            - 使用黑色书写的数字
            - 确保背景为白色
            - 保持数字清晰可见
            - 避免数字之间重叠
        """)
    
    # 主要内容区域
    if uploaded_file is not None:
        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 读取图片
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # 显示原始图片
            st.markdown("<h3 style='color: #1F2937'>📸 原始图片</h3>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='image-display'>", unsafe_allow_html=True)
                st.image(image, caption="上传的图片", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            progress_bar.progress(30)
            status_text.markdown("<p class='status-text'>🔄 正在处理图片...</p>", unsafe_allow_html=True)
            
            # 预处理
            processed_image = preprocess_image(image_array)
            st.markdown("<h3 style='color: #1F2937'>🔍 预处理结果</h3>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='image-display'>", unsafe_allow_html=True)
                st.image(processed_image, caption="预处理后的图片", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            progress_bar.progress(60)
            status_text.markdown("<p class='status-text'>✂️ 正在分割数字...</p>", unsafe_allow_html=True)
            
            # 分割数字
            digits = segment_digits(processed_image)
            
            if digits:
                st.markdown("<h3 style='color: #1F2937'>🎯 识别结果</h3>", unsafe_allow_html=True)
                progress_bar.progress(90)
                status_text.markdown("<p class='status-text'>🧠 正在进行智能识别...</p>", unsafe_allow_html=True)
                
                # 加载模型
                model = load_model()
                
                # 创建结果容器
                result_container = st.container()
                
                # 在容器中显示结果
                with result_container:
                    # 每行最多显示5个结果
                    for i in range(0, len(digits), 5):
                        cols = st.columns(min(5, len(digits) - i))
                        for j, col in enumerate(cols):
                            if i + j < len(digits):
                                digit = digits[i + j]
                                tensor = prepare_for_model(digit)
                                
                                with torch.no_grad():
                                    outputs = model(tensor)
                                    confidence, prediction = post_process_prediction(outputs, confidence_threshold)
                                    
                                    # 创建结果展示框
                                    col.markdown(
                                        f"""
                                        <div class='result-box'>
                                            <div style='text-align: center;'>
                                                <img src='data:image/png;base64,{base64.b64encode(cv2.imencode(".png", digit)[1]).decode()}' 
                                                     style='width: 80px; height: 80px; margin-bottom: 0.5rem;'>
                                                <div style='margin-top: 0.5rem;'>
                                                    {"<span class='confidence-high'>" if confidence.item() >= confidence_threshold else "<span class='confidence-low'>"}
                                                    <div style='font-size: 1.5rem; margin-bottom: 0.3rem;'>
                                                        {prediction.item() if confidence.item() >= confidence_threshold else "?"}
                                                    </div>
                                                    <div style='font-size: 0.9rem; color: #6B7280;'>
                                                        置信度: {confidence.item():.2f}
                                                    </div>
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                
                progress_bar.progress(100)
                status_text.markdown("<p class='status-text'>✨ 识别完成！</p>", unsafe_allow_html=True)
                
            else:
                progress_bar.progress(100)
                st.error("⚠️ 未检测到数字，请尝试上传其他图片")
                status_text.markdown("<p class='status-text'>❌ 处理完成，但未检测到数字</p>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"处理图片时出现错误: {str(e)}")
            status_text.markdown("<p class='status-text'>❌ 处理失败</p>", unsafe_allow_html=True)
    
    else:
        # 显示上传提示
        st.markdown(
            """
            <div class='upload-container'>
                <h3 style='color: #4B5563; margin-bottom: 1rem;'>👆 开始使用</h3>
                <p style='color: #6B7280;'>
                    在左侧面板上传手写数字图片，系统将自动进行识别
                </p>
                <div style='margin-top: 1rem; color: #9CA3AF; font-size: 0.9rem;'>
                    支持 PNG、JPG、JPEG 格式
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main() 