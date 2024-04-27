import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def edge_detection(image_path, threshold1=200, threshold2=290):
    """
    对输入的图像进行边缘检测，并显示结果。

    参数:
        image_path (str): 输入图像的文件路径。
        threshold1 (int): Canny 边缘检测的低阈值。
        threshold2 (int): Canny 边缘检测的高阈值。
    """
    img = cv.imread(image_path, 0)
    edges = cv.Canny(img, threshold1, threshold2)

    # 找到所有非零像素点的坐标
    indices = np.where(edges != 0)
    edge_points = np.column_stack((indices[1], indices[0]))  # 每行代表一个边缘点的坐标 (x, y)

    # 打印边缘点的坐标
    for point in edge_points:
        print(f"Edge point: ({point[0]}, {point[1]})")

    # 保存坐标到文件
    np.savetxt('edge_points.txt', edge_points, fmt='%d')


    plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# 测试封装的函数
edge_detection('image_2.jpeg')
