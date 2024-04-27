import cv2
import numpy as np

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 消噪
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 图像增强和对比度增强
    alpha = 1.5  # 对比度增强参数
    beta = 15  # 亮度增强参数
    enhanced_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=beta)

    # 高斯平滑
    blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    return blurred_image

def detect_quadrilaterals(image):
    # 图像预处理
    processed_image = preprocess_image(image)

    # 转换为灰度图像
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 复制图像以避免修改原始图像
    annotated_image = processed_image.copy()

    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        # 利用周长来估计轮廓的形状
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果逼近的轮廓有四个顶点，则认为是四边形
        if len(approx) == 4:
            # 计算四边形的面积
            area = cv2.contourArea(approx)
            # 放宽面积限制
            if area > 1000:  # 调整面积阈值
                # 在图像上绘制绿色的轮廓
                cv2.drawContours(annotated_image, [approx], -1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主函数
if __name__ == "__main__":
    image_path = 'zy2\image11.jpeg'
    detect_quadrilaterals(image_path)
