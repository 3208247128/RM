import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('image_2.jpeg', 0)
edges = cv.Canny(img, 100, 200)

# 寻找边缘点
indices = np.where(edges != 0)
points = np.column_stack((indices[1], indices[0]))  # 列表中的第一个元素是 x 坐标，第二个元素是 y 坐标

# 打印边缘点坐标
print("Edge points coordinates:")
for point in points:
    print(f"({point[0]}, {point[1]})")

# 使用拟合的直线方向向量计算单位向量
[vx, vy, x, y] = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
unit_vector = np.array([vy, -vx]) / np.linalg.norm([vx, vy])

# 计算直线起点和终点
top_point = (int(x + unit_vector[0] * 100), int(y + unit_vector[1] * 100))
bottom_point = (int(x - unit_vector[0] * 100), int(y - unit_vector[1] * 100))

# 在图像上绘制直线
color_img = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制彩色直线
cv.line(color_img, bottom_point, top_point, (0, 255, 0), 2)

plt.imshow(color_img)
plt.title('Edge Image with Detected Line'), plt.xticks([]), plt.yticks([])
plt.show()
