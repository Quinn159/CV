#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/16 22:09
# @Author   : kun
# @File    : main2.py

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time

start = time.clock()

# 提取图像水平、垂直线
img = cv.imread('C:\\Users\\HP\\Desktop\\test1\\07.2.bmp')
# img = cv.resize(img, (1024, 960), interpolation=cv.INTER_LINEAR)
# plt.imshow(img[:,:,-1])
# plt.show()
h, w = img.shape[:2]
xb, yb = int(0), int(h/2)       # 水平中心线与图像的交点
xc, yc = int(w), int(h/2)
# yb = int(h/2)   # 水平线方程

xs, ys = int(w/2), int(0)       # 垂直中心线与图像的交点
xt, yt = int(w/2), int(h)
# xs = int(w/2)   # 垂直线方程
# cv.line(img, (xb, yb), (xc, yc), (255, 0, 0), 3)
# cv.line(img, (xs, ys), (xt, yt), (255, 0, 0), 3)
# plt.imshow(img[:, :, ::-1])
# plt.show()

# 计算两条直线的交点的函数
def cross_point(line1, line2):
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

    # print(x1, y1, x2, y2)

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


# 图像预处理
# img = cv.imread('C:\\Users\\HP\\Desktop\\test1\\01.1.bmp')
# img = cv.resize(img, (960, 960), interpolation=cv.INTER_LINEAR)
# img = cv.imread('rect1.jpg')
# 转灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 高斯模糊
gray = cv.GaussianBlur(gray, (3, 3), 0)
# 边缘检测
edges = cv.Canny(gray, 60, 70)

# plt.imshow(edges, cmap='gray')
# plt.show()

# 概率霍夫变换
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=500)
lines1 = lines[:, 0, :]  # 提取为二维
# print(lines1)
for x1, y1, x2, y2 in lines1[:]:
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

lst1 = []
for x1, y1, x2, y2 in lines1:
    for x3, y3, x4, y4 in lines1:
        point_is_exist, [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
        if point_is_exist:
            if 0 < x < 1500 and 0 < y < 1800:       # 去除明显差异点，图像大小已知
                cv.circle(img, (int(x), int(y)), 9, (0, 0, 255), -1)
                # print((int(x), int(y)), end=',')
                x = int(x)
                y = int(y)
                lst1.append((x, y))
# print(list1, end=',')


# 得到四个顶点的坐标并依次排序
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
# print('ordered_points', ordered_points)
# plt.imshow(img[:, :, ::-1])
# plt.show()

x1 = ordered_points[0][0]
y1 = ordered_points[0][1]
x2 = ordered_points[1][0]
y2 = ordered_points[1][1]
x3 = ordered_points[2][0]
y3 = ordered_points[2][1]
x4 = ordered_points[3][0]
y4 = ordered_points[3][1]
# print('坐标', x1, y1, x2, y2, x3, y3, x4, y4)
# 计算水平，垂直线与矩形的交点
[xB_, yB_] = cross_point([x1, y1, x3, y3], [0, int(h / 2), int(w / 2), int(h / 2)])   # 计算是否存在交点，若存在则输出
[xC_, yC_] = cross_point([x2, y2, x4, y4], [0, int(h / 2), int(w / 2), int(h / 2)])
[xS_, yS_] = cross_point([x1, y1, x2, y2], [int(w/2), 0, int(w / 2), int(h / 2)])
[xT_, yT_] = cross_point([x3, y3, x4, y4], [int(w/2), 0, int(w / 2), int(h / 2)])
xB_, yB_ = [xB_, yB_][1][0], [xB_, yB_][1][1]
xC_, yC_ = [xC_, yC_][1][0], [xC_, yC_][1][1]
xS_, yS_ = [xS_, yS_][1][0], [xS_, yS_][1][1]
xT_, yT_ = [xT_, yT_][1][0], [xT_, yT_][1][1]
xD_, yD_ = int(w/2), int(h/2)
# print(xB, yB, xC, yC, xD, yD, xS, yS, xT, yT)

focalLength = 5600  # 焦距已由标定得到
# 相似关系计算B,C,D到A点的距离(两点间距离公式)
# w1 = w2 = w3 = 23       # 矩形垂直边和ST实际高度，单位cm
MQ = ST = NR = 23       # 矩形垂直边和ST实际高度，单位cm

D13 = math.sqrt(math.pow((x3 - x1), 2) + math.pow((y3 - y1), 2))
D24 = math.sqrt(math.pow((x4 - x2), 2) + math.pow((y4 - y2), 2))
DST = yT_ - yS_
# print(D13, D24, DST)
B_D_ = math.sqrt(math.pow((xD_ - xB_), 2) + math.pow((yD_ - yB_), 2))
AB_ = math.sqrt(math.pow(focalLength, 2) + math.pow(B_D_, 2))
AB = MQ * AB_/D13
# print('ABis', AB)
C_D_ = math.sqrt(math.pow((xC_ - xD_), 2) + math.pow((yC_ - yD_), 2))
AC_ = math.sqrt(math.pow(focalLength, 2) + math.pow(C_D_, 2))
AC = NR * AC_/D24
# print('ACis', AC)
BC = 28     # 矩形水平边实际长度


# dc = xC - xD    # DC的像素距离
# # print('dc', dc)
# bc = xC - xB    # BC的像素距离
#
# DC = dc/bc * BC     # DC实际长度
# # 计算夹角
angle_ABC = math.degrees(math.acos((AB*AB+BC*BC-AC*AC)/(2*AB*BC)))
BG = AB*B_D_/AB_
angle_ABG = math.degrees(math.acos(BG/AB))
# angle_DCA = math.degrees(math.acos((DC*DC+AC*AC-AD*AD)/(2*DC*AC)))
# angle_DAC = math.degrees(math.acos((AD*AD+AC*AC-DC*DC)/(2*AD*AC)))
# print('ABC,ABG的角度是：', angle_ABC, angle_ABG)  # 单位为角度制
theta = angle_ABC - angle_ABG
# angle_ECA = 90 - angle_DAC
# # print('ECA的角度是：', angle_ECA)
# angle_ECD = angle_DCA - angle_ECA
print('两平面的夹角是：', theta)
# # 可以得到目标平面相对相机平面在水平面内的夹角

end = time.clock()
print('Running time: %s Seconds' % (end-start))
