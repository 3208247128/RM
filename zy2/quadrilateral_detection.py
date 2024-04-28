import cv2
import numpy as np

def preprocess_image(image):
    # 消噪
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 图像增强和对比度增强
    alpha = 1.5  # 对比度增强参数
    beta = 15  # 亮度增强参数
    enhanced_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=beta)

    # 高斯平滑
    blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    return blurred_image

def find_quadrilaterals(processed_image):
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    quadrilaterals = []
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 排除面积较小的轮廓
        if area > 100:
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
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(marked_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(marked_image, f"({cx}, {cy})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return marked_image

# 测试
if __name__ == "__main__":
    image_path = 'zy2\image_2.jpeg'
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    quadrilaterals = find_quadrilaterals(processed_image)
    marked_image = mark_points_and_center(processed_image, quadrilaterals)
    cv2.imshow("Marked Image", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
