import cv2
import numpy as np

def preprocess_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 高斯平滑
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 灰度化
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化增强对比度
    equalized = cv2.equalizeHist(gray)
    
    return equalized

def detect_quadrilaterals(image_path):
    # 预处理图像
    preprocessed_img = preprocess_image(image_path)

    # 霍夫变换检测直线
    edges = cv2.Canny(preprocessed_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # 绘制检测到的直线
    img = cv2.imread(image_path)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 显示结果
    cv2.imshow('Lines Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 测试函数
detect_quadrilaterals('zy2\image_2.jpeg')
