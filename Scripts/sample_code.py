import cv2
import numpy as np
from sklearn.cluster import KMeans

# 이미지 파일을 읽어온다.
img_file = './Resources/road_2.jpg'
img = cv2.imread(img_file)

# 그레이 스케일로 변환한다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 블러처리 한다.
gray = cv2.medianBlur(gray, 5)

# 엣지 검출을 수행한다.
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

# 컬러화한다.
color = cv2.bilateralFilter(img, 9, 250, 250)

# 카툰 렌더링을 수행한다.
cartoon = cv2.bitwise_and(color, color, mask=mask)

# 결과를 출력한다.
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()