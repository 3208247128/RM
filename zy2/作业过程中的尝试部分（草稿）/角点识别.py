import cv2
import numpy as np

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
    blurred_image = cv2.GaussianBlur(denoised_image, (3, 3), 0)

    return blurred_image

def detect_and_draw_corners(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测角点
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.intp(corners)

    # 绘制角点
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    return image

# 主函数
if __name__ == "__main__":
    image_path = 'zy2\image_2.jpeg'

    # 图像预处理
    processed_image = preprocess_image(image_path)

    # 检测并绘制角点
    corner_image = detect_and_draw_corners(processed_image)

    # 显示结果
    cv2.imshow('Corners Detected Image', corner_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
