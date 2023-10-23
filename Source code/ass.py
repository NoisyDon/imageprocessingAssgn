import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt


#folders
filePath = 'D:/Users/Alexg/Desktop/MultanT2MRI/Healthy/'
#filePath = 'D:/Users/Alexg/Desktop/MultanT2MRI/HERNIATED/'

images = []
for filename in os.listdir(filePath):
    img = cv.imread(os.path.join(filePath,filename))
    if img is not None:
        images.append(img)

for i in range(len(images)):
    im = images[i]

    """ #tried to use template but not working that well
    #template image
    template_im = cv.imread('D:/Users/Alexg/Desktop/MultanT2MRI/template.jpg')

    template_im = cv.GaussianBlur(template_im, (3, 3), 0)
    res = cv.matchTemplate(im,template_im,cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    cv.imshow('template',template_im)

    w, h, layer = template_im.shape
    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)

    im = im[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.medianBlur(im, 3)
    im = cv.equalizeHist(im)

    cv.imshow('after equalization' + str(i), im)

    im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

    im = cv.GaussianBlur(im, (3, 3), 0)

    #for the lower back
    lowVal = 11.5/100*255       #45
    highVal = 29.5/100*255      #65
    lowRange = np.array([[0,0,lowVal]]) #10 -20, max 100
    highRange = np.array([[0,0,highVal]]) #but since opencv use 0-255, need normalize

    roi = cv.inRange(im, lowRange, highRange)
    ret, roi = cv.threshold(roi,127,255,cv.THRESH_BINARY)

    maskTest = np.zeros(roi.shape,np.uint8)
    maskSmall = np.zeros(roi.shape,np.uint8)
    maskMed = np.zeros(roi.shape,np.uint8)
    maskBig = np.zeros(roi.shape,np.uint8)

    contours, hier = cv.findContours(roi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #if 10<cv.contourArea(cnt)<300:
        if 5<cv.contourArea(cnt)<300:
            cv.drawContours(roi,[cnt],0,255,cv.FILLED)
            cv.drawContours(maskTest,[cnt],0,255,-1)

    #get small elements
    contours, hier = cv.findContours(roi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 15<cv.contourArea(cnt)<30:
            cv.drawContours(roi,[cnt],0,255,cv.FILLED)
            cv.drawContours(maskSmall,[cnt],0,255,-1)

    #get med elements
    contours, hier = cv.findContours(roi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 25<cv.contourArea(cnt)<55:
            cv.drawContours(roi,[cnt],0,255,cv.FILLED)
            cv.drawContours(maskMed,[cnt],0,255,-1)

    #get bigger elements #perfected for healthy first pic
    contours, hier = cv.findContours(roi,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 70<cv.contourArea(cnt)<80:
            cv.drawContours(roi,[cnt],0,255,cv.FILLED)
            cv.drawContours(maskBig,[cnt],0,255,-1)

    finalMask = cv.bitwise_or(maskSmall,maskMed)
    finalMask = cv.bitwise_or(finalMask,maskBig)

    im1 = cv.Canny(finalMask, 40, 50)
    #cv.imshow('Canny' + str(i), im1)
    cv.imshow('Test' + str(i), roi)
    cv.imshow('mask test' + str(i), maskTest)
    #cv.imshow('mask small' + str(i), maskSmall)
    #cv.imshow('mask med' + str(i), maskMed)
    #cv.imshow('mask big' + str(i), maskBig)
    #cv.imshow('Final' + str(i), finalMask)
    cv.waitKey(0) """


    count = 1
    rows = 5
    column = 1

    print(i)

    im = im[40:200,60:140]

    #using range after CLAHE
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    cl1 = clahe.apply(im)

    lookUpTable = np.empty((1,256), np.uint8)
    for x in range(256):
        lookUpTable[0,x] = np.clip(pow(x / 255.0, 0.8) * 255.0, 0, 255)
    res = cv.LUT(cl1, lookUpTable)

    temp = cv.cvtColor(res, cv.COLOR_GRAY2BGR)
    temp = cv.cvtColor(temp, cv.COLOR_BGR2HSV)

    h,s,v = cv.split(temp)

    blur = cv.GaussianBlur(v,(5,5),0)
    th = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 3, 0)

    contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    drawing = np.zeros((th.shape[0], th.shape[1]), dtype=np.uint8)
    for x in range(len(contours)):
        cv.drawContours(drawing, contours, x, (255, 255, 255), cv.FILLED, cv.LINE_8, hierarchy, 0)

    invert = cv.bitwise_not(drawing)

    row_count = 0

    # plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(v, cmap='gray')
    # row_count +=1
    # plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(blur, cmap='gray')
    # row_count +=1
    # plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(th, cmap='gray')
    # row_count +=1
    # plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(drawing, cmap='gray')
    # row_count +=1
    # plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(invert, cmap='gray')


    count += 1

    cv.namedWindow('Selection of ROI', cv.WINDOW_KEEPRATIO)
    boxes = cv.selectROIs('Selection of ROI', invert)

    count_in = 1
    rows_in = 10
    column_in = 5

    print(i)

    for rect in boxes:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        im_roi = invert[y1:y1+y2,x1:x1+x2]

        row_count_inner = 0

    
    
        plt.figure(str(i)+'boxes'),plt.subplot(rows_in,column_in,count_in+column_in*row_count_inner),plt.imshow(im_roi, cmap='gray')
        count_in += 1


plt.show()
cv.waitKey(0)