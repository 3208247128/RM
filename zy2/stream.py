import cv2
import redgreen
from quadrilateral_detection import find_quadrilaterals, mark_points_and_center



def detect_and_mark(image_path):
    # 检测红色和绿色区域并标记
    processed_image = redgreen.detect_red_and_green(image_path)

    # 读取图像并检测四边形
    image = cv2.imread(image_path)
    quadrilaterals = find_quadrilaterals(image)

    # 标记顶点和中心点
    marked_image = mark_points_and_center(image, quadrilaterals)

    # 将两个标记合成一个图像
    combined_image = cv2.addWeighted(processed_image, 0.5, marked_image, 0.5, 0)

    return combined_image

# 主函数
if __name__ == "__main__":
    image_path = 'zy2\image_1.jpeg'

    combined_image = detect_and_mark(image_path)

    # 显示合成的图像
    cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
