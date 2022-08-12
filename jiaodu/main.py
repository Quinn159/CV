#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/5 16:42
# @Author   : kun
# @File    : main.py


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# 初始化从摄像机到物体的已知距离
KNOWN_DISTANCE = 14

# 初始化已知的对象宽度，A4纸的长和宽(单位:inches)
KNOWN_WIDTH = 8.26
KNOWN_HEIGHT = 5.90

def find_marker(image):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    edged_img = cv.Canny(gray_img, 35, 125)
    img, countours, hierarchy = cv.findContours(edged_img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # 提取最大面积矩形对应的点集
    c = max(countours, key=cv.contourArea)
    # cv2.minAreaRect()函数返回矩形的中心点坐标，长宽，旋转角度[-90,0)，当矩形水平或竖直时均返回-90
    rect = cv.minAreaRect(c)
    return rect

# 计算焦距
def calculate_focalDistance(img_path):
    first_image = cv.imread(img_path)
    # 得到最小外接矩形的中心点坐标，长宽，旋转角度
    # 其中marker[1][0]是该矩形的宽度，单位为像素
    marker = find_marker(first_image)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    # print('焦距（focalLength ）= ', focalLength)
    return focalLength

if __name__ == "__main__":
    img_path = "image0(1).jpg"
    focalLength = calculate_focalDistance(img_path)


# 提取图像水平、垂直线
img = cv.imread('cizhuan(1).jpg')
h, w = img.shape[:2]
xb, yb = int(0), int(h/2)       # 水平中心线与图像的交点
xc, yc = int(w), int(h/2)
cv.line(img, (xb, yb), (xc, yc), (255, 0, 0), 3)

xs, ys = int(w/2), int(0)       # 垂直中心线与图像的交点
xt, yt = int(w/2), int(h)
cv.line(img, (xs, ys), (xt, yt), (255, 0, 0), 3)


# 获取线段的像素长度
def cross_point(line1, line2):  # 计算交点函数
    #是否存在交点
    point_is_exist = False
    x = 0
    y = 0
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]


# 确定两条直线的斜率
    if (x2 - x1) == 0:      # 垂直线,L1斜率不存在
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist=True     # 交点x,y存在
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist=True
    return point_is_exist, [x, y]
# 图片路径
img = cv.imread('cizhuan(1).jpg')
# 转灰度图
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 高斯模糊
gray = cv.GaussianBlur(gray, (3, 3), 0)
# 边缘检测
edges = cv.Canny(gray, 100, 200)


# 概率霍夫变换
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=300, maxLineGap=300)
lines1 = lines[:, 0, :]  # 提取为二维
lst1 = []
for x1, y1, x2, y2 in lines1:
    for x3, y3, x4, y4 in lines1:
        point_is_exist, [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
        if point_is_exist:
            if 0 < x < 1000 and 0 < y < 1000:
                # cv.circle(img, (int(x), int(y)), 15, (0, 0, 255), -1)
                # print((int(x), int(y)), end=',')
                x = int(x)
                y = int(y)
                lst1.append((x, y))


points = np.array(lst1)
ordered_points = np.zeros([4, 2])
# 将横纵坐标相加,最小为左上角,最大为右下角
add = np.sum(points, axis=1)
ordered_points[0] = points[np.argmin(add)]
ordered_points[3] = points[np.argmax(add)]
# 将横纵坐标相减 diff 为后减前 即 y-x,最小为右上角,最大为左下角
diff = np.diff(points, axis=1)
ordered_points[1] = points[np.argmin(diff)]
ordered_points[2] = points[np.argmax(diff)]


x1 = ordered_points[0][0]
y1 = ordered_points[0][1]
x2 = ordered_points[1][0]
y2 = ordered_points[1][1]
x3 = ordered_points[2][0]
y3 = ordered_points[2][1]
x4 = ordered_points[3][0]
y4 = ordered_points[3][1]


focalLength = 232.20804505429027  # 焦距已单独求出
# 计算B,C,D到A点的距离
w1 = w2 = 45       # 矩形垂直边和ST实际高度，单位cm
w3 = 50
AB = w1 * focalLength/(y3-y1)
AC = w2 * focalLength/(y4-y2)
# 图像中心点D坐标（200,150）
AD = w3 * focalLength/(x2-x1)


[xB, yB] = cross_point([x1, y1, x3, y3], [0, int(h / 2), int(w / 2), int(h / 2)])   # 计算是否存在交点，若存在则输出
[xC, yC] = cross_point([x2, y2, x4, y4], [0, int(h / 2), int(w / 2), int(h / 2)])
xB, yB = [xB, yB][1][0], [xB, yB][1][1]
xC, yC = [xC, yC][1][0], [xC, yC][1][1]
xD, yD = int(w/2), int(h/2)

dc = xC - xD    # DC的像素距离
bc = xC - xB    # BC的像素距离
BC = 45   # BC的实际长度
DC = dc/bc * BC     # DC实际长度
# 计算夹角
angle_DCA = math.degrees(math.acos((DC*DC+AC*AC-AD*AD)/(2*DC*AC)))
angle_DAC = math.degrees(math.acos((AD*AD+AC*AC-DC*DC)/(2*AD*AC)))
print('DCA,DAC的角度是：', angle_DCA, angle_DAC)  #单位为角度制
angle_ECA = 90 - angle_DAC
print('ECA的角度是：', angle_ECA)
angle_ECD = angle_DCA - angle_ECA
print('ECD的角度是：', angle_ECD)
# 可以得到目标平面相对相机平面在水平面内的夹角∠ECD
#
#
