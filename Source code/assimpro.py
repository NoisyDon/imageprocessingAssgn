import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

#folders
#filePath = 'D:/Users/Alexg/Desktop/MultanT2MRI/Healthy/'
filePath = 'D:/Users/Alexg/Desktop/MultanT2MRI/HERNIATED/'

images = []
for filename in os.listdir(filePath):
    img = cv.imread(os.path.join(filePath,filename))
    if img is not None:
        images.append(img)

for i in range(len(images)):
    im = images[i]

    cv.namedWindow('Selection of ROI', cv.WINDOW_KEEPRATIO)
    boxes = cv.selectROIs('Selection of ROI', im)

    count = 1
    rows = 15
    column = 5
    print(boxes)

    for rect in boxes:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        im_roi = im[y1:y1+y2,x1:x1+x2]
        im_roi = cv.resize(im_roi, dsize=None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        im_roi = cv.cvtColor(im_roi, cv.COLOR_BGR2GRAY)
        #file:///D:/Users/Alexg/Desktop/fk/2015_Comput_Biol_Med_2015_62_196_205.pdf
        #equalization 

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(im_roi)

        #threholdingusing otsu, dont work
        #ret,th = cv.threshold(im_roi,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        lookUpTable = np.empty((1,256), np.uint8)
        for x in range(256):
            lookUpTable[0,x] = np.clip(pow(x / 255.0, 0.8) * 255.0, 0, 255) #ori 1.2
        res = cv.LUT(cl1, lookUpTable)

        #normal thresholding using range
        #for the black ring around each disc
        temp = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        temp = cv.cvtColor(temp,cv.COLOR_BGR2HSV)
        lowVal = 0/100*255 
        highVal = 50/100*255 #35, 40 when gamma 1.2, 50 when gamme 0.8 n no blur, 55, 0.8 when need subtract
        lowRange = np.array([[0,0,lowVal]]) #10 -20, max 100
        highRange = np.array([[0,0,highVal]]) #but since opencv use 0-255, need normalize
        roi = cv.inRange(temp, lowRange, highRange)
        ret,th = cv.threshold(roi,0,255,cv.THRESH_BINARY)

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(th, connectivity=4)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        #min_size = 75  
        #keep only largest component
        min_size = 0
        for a in range(0, nb_components):
            if sizes[a] > min_size:
                min_size = sizes[a]

        #your answer image
        img2 = np.zeros((output.shape), np.uint8)
        #for every component in the image, you keep it only if it's above min_size
        for j in range(0, nb_components):
            if sizes[j] >= min_size:
                img2[output == j + 1] = 255

        thin = cv.ximgproc.thinning(img2)
        """ contours, hierarchy = cv.findContours(img2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        thin = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
        for x in range(len(contours)):
            #draw outline
            #cv.drawContours(thin, contours, -1, (255,255,255), 1)
            cv.drawContours(thin, contours, x, (255, 255, 255), cv.FILLED, cv.LINE_8, hierarchy, 0)

        se = cv.getStructuringElement(cv.MORPH_RECT, (3,4))
        di_im = cv.dilate(thin,se,iterations=1)
        er_im = cv.erode(di_im,se,iterations=1) """


        #no need if box is big
        #to connect
        #kernel = np.ones((1,19), np.uint8)
        kernel = np.array([[0,0,0,0,0],
        [0,1,0,1,0],
        [1,1,1,1,1],
        [0,1,0,1,0],
        [0,0,0,0,0]], dtype=np.uint8)
        di_im = cv.dilate(thin, kernel, iterations=2)
        #er_im = cv.erode(di_im, kernel, iterations=2)
        er_im = cv.ximgproc.thinning(di_im)

        # Find contours
        contours, hierarchy = cv.findContours(er_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # Draw contours
        drawing1 = np.zeros((th.shape[0], th.shape[1]), dtype=np.uint8)

        area = 0

        for x in range(len(contours)):
            cnt = contours[0]
            #onyl fill in largest contour area
            if cv.contourArea(cnt) > area:
                area = cv.contourArea(cnt)

        for x in range(len(contours)):
            #onyl fill in largest contour area
            if cv.contourArea(cnt) == area:  
                cv.drawContours(drawing1, contours, x, (255, 255, 255), cv.FILLED, cv.LINE_8, hierarchy, 0)

        #remove shit noise
        noise = cv.medianBlur(drawing1, 5)

        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(noise, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes1 = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        #keep only largest component
        min_size = 0
        for a in range(0, nb_components):
            if sizes1[a] > min_size:
                min_size = sizes1[a]

        #your answer image
        img3 = np.zeros((output.shape), np.uint8)
        #for every component in the image, you keep it only if it's above min_size
        for j in range(0, nb_components):
            if sizes1[j] >= min_size:
                img3[output == j + 1] = 255

        ####
        '''
        #gaussian
        gaussian = cv.GaussianBlur(th, (5, 5), 0)
        '''
        #gradient vector flow, used single seed point
        row_count = 0

        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(im_roi, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(cl1, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(res, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(temp, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(th, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(img2, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(thin, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(di_im, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(er_im, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(drawing1, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(noise, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(img3, cmap='gray')


        count += 1
        
plt.show()
cv.waitKey(0)
print('No error')