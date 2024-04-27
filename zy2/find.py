import cv2
import numpy as np

# 导入红绿色检测函数
from redgreen import detect_red_and_green

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 图像增强和对比度增强
    alpha = 1.5  # 对比度增强参数
    beta = 15  # 亮度增强参数
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 去噪
    denoised_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)

    # 高斯平滑
    blurred_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)

    return blurred_image

def detect_quadrilaterals(image):
    # 边缘检测
    edges = cv2.Canny(image, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制图像以避免修改原始图像
    annotated_image = image.copy()

    # 遍历轮廓
    for contour in contours:
        # 进行轮廓逼近
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 如果逼近的轮廓有四个顶点，则认为是四边形
        if len(approx) == 4:
            # 在图像上绘制绿色的轮廓
            cv2.drawContours(annotated_image, [approx], -1, (0, 255, 0), 2)

            # 计算四边形的中心点
            M = cv2.moments(approx)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # 在图像上标记中心点为蓝色
                cv2.circle(annotated_image, (cx, cy), 5, (255, 0, 0), -1)

            # 在图像上标记四个顶点并打印坐标
            for point in approx:
                x, y = point[0]
                cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(annotated_image, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return annotated_image

# 主函数
if __name__ == "__main__":
    image_path = 'zy2\image_2.jpeg'

    # 图像预处理
    processed_image = preprocess_image(image_path)

    # 检测四边形并标记特征
    annotated_image_quadrilaterals = detect_quadrilaterals(processed_image)

    # 检测红色和绿色区域并标记
    red_green_image = detect_red_and_green(image_path)

    # 将两个图像叠加起来
    combined_image = cv2.addWeighted(annotated_image_quadrilaterals, 0.5, red_green_image, 0.5, 0)

    # 显示结果
    cv2.imshow('Combined Annotated Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
