# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:40:37 2020

@author: zxuzhi
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os 

os.chdir('../coloring')





import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame



i=8
if i<10:
    filename = 'coloring-0' +str(i)+'.png'
else:
    filename = 'coloring-' +str(i)+'.png'
img = cv2.imread(filename)
#plt.imshow(img)
#plt.show()
img = img.copy()
img[img<=248] = 0 
#plt.imshow(img)
#plt.show()


grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 200, 255), 2)
#plt.imshow(img)
#plt.show()
hierarchyDF = DataFrame(hierarchy[0], columns = ['pre', 'next', 'child', 'parent'])

#area = cv2.contourArea(cnt)
#area_size = fabs(cv.contourArea(cnt))

#contruct dictionary of parent: child relation
contour_dic = dict()
for i in range(0,hierarchyDF.shape[0]):
    parent_cnt_num =  hierarchyDF.iat[i,3]
    child_cnt_num =i
    if parent_cnt_num in contour_dic:
        contour_dic[parent_cnt_num].append(child_cnt_num)
    else:
        contour_dic[parent_cnt_num] = [child_cnt_num]


#find parent node
cont_lst = []
for i in range(0,len(contours)):
    cont_lst.append(i)
for thekey in contour_dic.keys():
    if thekey!=-1:
        for item in contour_dic[thekey]:
            cont_lst.remove(item)
    

#BFS all the nodes
while len(cont_lst)>0:
    if cont_lst[0] in contour_dic.keys():
        
        parent_cnt_num = cont_lst[0]
        child_cnt_num_lst= contour_dic[parent_cnt_num]
        
        cnt =list()
        cnt.append(contours[parent_cnt_num].copy())
        
        for thecntnum in child_cnt_num_lst:
            cnt.append(contours[thecntnum].copy())
        #moment = cv2.moments(cnt)
        #c_y = moment['m10']/(moment['m00']+0.01)
        #c_x = moment['m01']/(moment['m00']+0.01)
        #centroid_color = img[c_x,c_y]
        #centroid_color = np.array((centroid_color[0],centroid_color[1],centroid_color[2]))
        r1 = random.randint(0,255)
        r2= random.randint(0,255)
        r3 = random.randint(0,255)
        #area = cv2.contourArea(cnt)

        color = np.uint8(np.random.rand(3) * 255).tolist()

        img = cv2.fillPoly(img, cnt,color)
        cont_lst.pop(0)
        cont_lst.extend(child_cnt_num_lst)
    else:
        cnt = contours[cont_lst[0]]
        print(cont_lst[0])

        color = np.uint8(np.random.rand(3) * 255).tolist()

        
        img = cv2.fillPoly(img, cnt,color)
        cont_lst.pop(0)
        
    
#plt.imshow(img)
#plt.show()

cv2.imshow('i', img)
cv2.waitKey(0)
if i<10:
    filename = 'coloring-0' +str(i)+'_processed.png'
else:
    filename = 'coloring-' +str(i)+'_processed.png'
cv2.imwrite('processed_5683.png', img) 



import math
rows, cols = img.shape[:2]


centerX = rows / 2
centerY = cols / 2

radius = min(centerX, centerY)



strength = 200


dst = np.zeros((rows, cols, 3), dtype="uint8")


for i in range(rows):
    for j in range(cols):
  
        distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
     
        B =  img[i,j][0]
        G =  img[i,j][1]
        R = img[i,j][2]
        if (distance < radius * radius):
      
            result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
            B = img[i,j][0] + result
            G = img[i,j][1] + result
            R = img[i,j][2] + result

            B = min(255, max(0, B))
            G = min(255, max(0, G))
            R = min(255, max(0, R))
            dst[i,j] = np.uint8((B, G, R))
        else:
            dst[i,j] = np.uint8((B, G, R))
        

cv2.imshow('src', img)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('processed_56832.png', dst) 




rows, cols = img.shape[:2]


dst = np.zeros((rows, cols, 3), dtype="uint8")


for i in range(rows):
    for j in range(cols):
        B = 0.272*img[i,j][2] + 0.534*img[i,j][1] + 0.131*img[i,j][0]
        G = 0.349*img[i,j][2] + 0.686*img[i,j][1] + 0.168*img[i,j][0]
        R = 0.393*img[i,j][2] + 0.769*img[i,j][1] + 0.189*img[i,j][0]
        if B>255:
            B = 255
        if G>255:
            G = 255
        if R>255:
            R = 255
        dst[i,j] = np.uint8((B, G, R))
        

cv2.imshow('src', img)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('processed_56833.png', dst) D