import cv2
import canny_detect
import redgreen

def detect_and_mark(image_path):
    # 使用canny_detect模块进行图像处理和四边形检测
    processed_image = canny_detect.preprocess_image(cv2.imread(image_path))
    quadrilaterals = canny_detect.find_quadrilaterals(processed_image)
    marked_image = canny_detect.mark_points_and_center(cv2.imread(image_path), quadrilaterals)
    
    # 使用redgreen模块检测红色和绿色区域并标记
    red_green_marked_image = redgreen.detect_red_and_green(image_path)

    # 将两个标记合并到一张图像上
    combined_image = cv2.addWeighted(marked_image, 0.5, red_green_marked_image, 0.5, 0)

    return combined_image

# 主函数
if __name__ == "__main__":
    image_path = 'image_1.jpeg'  # 请替换为您的图像路径
    combined_image = detect_and_mark(image_path)

    # 显示合成的图像
    cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
