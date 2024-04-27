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

def detect_quadrilaterals(image_path, iterations=3):
    # 图像预处理
    processed_image = preprocess_image(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 复制图像以避免修改原始图像
    annotated_image = processed_image.copy()

    # 初始化轮廓逼近的阈值
    epsilon = 0.02

    # 初始化中心点
    center = (processed_image.shape[1] // 2, processed_image.shape[0] // 2)

    # 开始轮廓逼近，直到中心点
    for contour in contours:
        # 逼近轮廓
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

        # 绘制逼近的四边形
        cv2.drawContours(annotated_image, [approx], -1, (0, 0, 255), 2)

        # 更新轮廓逼近的阈值
        epsilon += 0.001

        # 如果中心点在逼近的四边形内，则放大四边形并再次进行识别
        if cv2.pointPolygonTest(approx, center, False) >= 0:
            # 多次进行放大和识别
            for _ in range(iterations):
                # 放大四边形
                approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

                # 绘制逼近的四边形
                cv2.drawContours(annotated_image, [approx], -1, (0, 255, 0), 2)

                # 更新轮廓逼近的阈值
                epsilon += 0.001

                # 如果中心点在逼近的四边形内，则继续放大和识别
                if cv2.pointPolygonTest(approx, center, False) < 0:
                    break

    # 显示结果
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主函数
if __name__ == "__main__":
    image_path = 'zy2\image_2.jpeg'
    detect_quadrilaterals(image_path)
