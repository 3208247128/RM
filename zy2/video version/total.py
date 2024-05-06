import cv2
import numpy as np

def detect_red_and_green_and_mark_quadrilaterals(camera_index=1):
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            break

        # 将图像从 BGR 色彩空间转换为 HSV 色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # 在图像上标出红色区域的中心坐标
            cv2.putText(frame, f'Red: ({x + w // 2}, {y + h // 2})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 遍历绿色区域的轮廓
        for contour in contours_green:
            # 获取轮廓的边界框坐标
            x, y, w, h = cv2.boundingRect(contour)
            # 在图像上绘制绿色边界框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 在图像上标出绿色区域的中心坐标
            cv2.putText(frame, f'Green: ({x + w // 2}, {y + h // 2})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 预处理图像并标记四边形
        processed_image = preprocess_image(frame)
        quadrilaterals = find_quadrilaterals(processed_image)
        marked_image = mark_points_and_center(frame, quadrilaterals)

        # 显示图像
        cv2.imshow('Marked Image', marked_image)

        # 如果按下 q 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image):
    # 消噪
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 图像增强和对比度增强
    alpha = 1.0  # 对比度增强参数
    beta = 15  # 亮度增强参数
    enhanced_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=beta)

    # 边缘检测
    edges = cv2.Canny(enhanced_image, 50, 150)

    return edges

def find_quadrilaterals(processed_image):
    # 查找轮廓
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    quadrilaterals = []
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 排除面积较小的轮廓
        if area > 2000:
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)

            # 对轮廓进行多边形逼近
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # 如果逼近的轮廓是四边形，将其添加到列表中
            if len(approx) == 4:
                quadrilaterals.append(approx)

    return quadrilaterals

def mark_points_and_center(image, quadrilaterals):
    marked_image = image.copy()

    for quad in quadrilaterals:
        # 绘制轮廓
        cv2.drawContours(marked_image, [quad], -1, (255, 0, 0), 2)

        # 绘制顶点并标记坐标
        for i, point in enumerate(quad):
            x, y = point[0]
            cv2.circle(marked_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(marked_image, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 计算中心点并标记坐标
        M = cv2.moments(quad)
        if M['m00'] != 0:  # 避免除以零错误
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(marked_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(marked_image, f"({cx}, {cy})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return marked_image

if __name__ == "__main__":
    detect_red_and_green_and_mark_quadrilaterals()
