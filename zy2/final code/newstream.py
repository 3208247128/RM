import cv2
import canny_detect
import redgreen

def detect_mark_and_save(image_path, output_path):
    # 预处理图像并检测四边形
    processed_image = canny_detect.preprocess_image(cv2.imread(image_path))
    quadrilaterals = canny_detect.find_quadrilaterals(processed_image)

    # 标记顶点和中心点
    marked_image = canny_detect.mark_points_and_center(cv2.imread(image_path), quadrilaterals)

    # 检测红色和绿色区域并标记
    processed_redgreen_image = redgreen.detect_red_and_green(image_path)

    # 将两个标记合成一个图像
    combined_image = cv2.addWeighted(processed_redgreen_image, 0.5, marked_image, 0.5, 0)

    # 保存合成的图像
    cv2.imwrite(output_path, combined_image)

if __name__ == "__main__":
    image_path = r'src\image11.jpeg'
    output_path = r'D:\githubjjz\J\J\zy2\out\combined_image3.jpg'
    
    # 调用函数处理图像并保存
    detect_mark_and_save(image_path, output_path)

    # 显示合成的图像
    combined_image = cv2.imread(output_path)
    cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
