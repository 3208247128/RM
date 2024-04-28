import cv2
import numpy as np

def detect_red_and_green(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从 BGR 色彩空间转换为 HSV 色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色和绿色的HSV范围
    lower_red = np.array([165, 100, 100])
    upper_red = np.array([179, 255, 255])
    lower_green = np.array([36, 100, 100])
    upper_green = np.array([86, 255, 255])

    # 根据阈值创建红色和绿色的掩码
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)



    # 寻找红色和绿色区域的轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历红色区域的轮廓
    for contour in contours_red:
        # 获取轮廓的边界框坐标
        x, y, w, h = cv2.boundingRect(contour)
        # 在图像上绘制红色边界框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # 在图像上标出红色区域的中心坐标
        cv2.putText(image, f'Red: ({x + w // 2}, {y + h // 2})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 遍历绿色区域的轮廓
    for contour in contours_green:
        # 获取轮廓的边界框坐标
        x, y, w, h = cv2.boundingRect(contour)
        # 在图像上绘制绿色边界框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 在图像上标出绿色区域的中心坐标
        cv2.putText(image, f'Green: ({x + w // 2}, {y + h // 2})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 返回带有标记的图像
    return image


if __name__ == "__main__":
    image_path = 'src\image_2.jpeg'
    combined_image = detect_red_and_green(image_path)

    # 显示合成的图像
    cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()