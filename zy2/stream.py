import redgreen
import cv2
import edge_detection

# 检测红色和绿色区域并标记
processed_image = redgreen.detect_red_and_green('zy2\image_2.jpeg')

# 显示带有标记的图像
cv2.imshow('Detected Colors', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge_detection('zy2\image_2.jpeg')



