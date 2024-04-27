import cv2
import numpy as np

def preprocess_image(image):
    # 高斯平滑
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 增强对比度
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=0)
    
    # 二值化
    _, thresholded = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)

    return thresholded

def detect_quadrilaterals(image):
    # 轮廓检测
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制四边形
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 如果逼近多边形具有四个顶点，则绘制方框
        if len(approx) == 4:
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    return image

def main():
    # 读取灰度图像
    gray_image = cv2.imread('zy2\屏幕截图 2024-04-27 161129.jpeg', cv2.IMREAD_GRAYSCALE)

    # 图像预处理
    processed_image = preprocess_image(gray_image)

    # 在预处理后的图像上识别四边形并绘制
    result_image_with_quadrilaterals = detect_quadrilaterals(processed_image)

    # 显示结果
    cv2.imshow('Detected Quadrilaterals', result_image_with_quadrilaterals)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
