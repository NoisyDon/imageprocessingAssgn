import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size

#folders
filePath = 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/Healthy/'
#filePath = 'D:/Users/Alexg/Desktop/MultanT2MRI/HERNIATED/'

#for final results
result_roi = []
result_segment = []
row_res = 22

#get all the images from the folder and put together as an array
images = []
for filename in os.listdir(filePath):
    img = cv.imread(os.path.join(filePath,filename))
    if img is not None:
        images.append(img)

#loop through all the images
for i in range(len(images)):
    im = images[i]
    
    #lets the user draw the ROI of the discs out
    cv.namedWindow('Selection of ROI', cv.WINDOW_KEEPRATIO)
    boxes = cv.selectROIs('Selection of ROI', im)

    #for the plotting later on
    count = 1
    rows = 15
    column = 5

    #for debuging
    print(i)
    print(boxes)

    #initializes the counts for saving images
    im_count = 0

    #goes though each ROI by the user
    for rect in boxes:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        #crops the image to the ROI drawn
        im_ori = im[y1:y1+y2,x1:x1+x2]

        #resizes the image by scaling up by 2 in each axis using cubic interpolation
        im_roi = cv.resize(im_ori, dsize=None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        #opencv reads images in BGR, change BGR to grayscale
        im_roi = cv.cvtColor(im_roi, cv.COLOR_BGR2GRAY)

        #form a CLAHE mask and apply it on the ROI drawn, where the mask is 4x4 large
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(im_roi)

        #perform gamma transformation on the image
        lookUpTable = np.empty((1,256), np.uint8)
        for x in range(256):
            lookUpTable[0,x] = np.clip(pow(x / 255.0, 0.8) * 255.0, 0, 255)
        res = cv.LUT(cl1, lookUpTable)

        #change the image from grayscale to HSV
        temp = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        temp = cv.cvtColor(temp,cv.COLOR_BGR2HSV)

        #only get the V component
        h,s,v = cv.split(temp)

        #set lowest and highest threshold to get the pixels in range of these values
        lowVal = 0
        highVal = 120
        roi = cv.inRange(v, lowVal, highVal)
        ret,th = cv.threshold(roi,0,255,cv.THRESH_BINARY)

        #create a black border around the image 10 pixels wide to remove noise near border easily
        bordered = cv.copyMakeBorder(th, 10,10,10,10,cv.BORDER_CONSTANT, 0)

        #find all connected components (white blobs in image)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(bordered, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        #sort the list of sizes from largest value to smallest value and then add a 0 behind to ease calculation later on
        sizes_sorted = sorted(sizes, reverse=True)
        sizes_sorted.append(0)
        
        #compare the difference in value of each size, only keeps blobs with big difference with next value
        for x in range(0, len(sizes_sorted)):
            if len(sizes_sorted) == 1:
                print('no comparison, only 1 data')
                min_size = sizes_sorted[x]
            else:
                if (sizes_sorted[x-1]-sizes_sorted[x])>15:
                    if sizes_sorted[x-1] >= 80:
                        min_size = sizes_sorted[x-1]

        #create empty image to move big blobs in
        conn_comp = np.zeros((output.shape), np.uint8)
        #for every component in the image, you keep it only if it's above min_size
        for x in range(0, nb_components):
            if sizes[x] >= min_size:
                conn_comp[output == x + 1] = 255

        #blur the image using median blur to remove more noise (the 4 corners of each blob)
        noise_conn_comp = cv.medianBlur(conn_comp,3)

        #create straight 1x3 structuing element and apply erosion and dilation
        se = np.ones((1,3))
        morph_after_noise = cv.erode(noise_conn_comp,se)
        morph_after_noise = cv.dilate(morph_after_noise,se)

        #find all connected components again and remove smaller blobs
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(morph_after_noise, connectivity=4)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        sizes_sorted = sorted(sizes, reverse=True)
        sizes_sorted.append(0)
        
        for x in range(0, len(sizes_sorted)):
            if len(sizes_sorted) == 1:
                min_size = sizes_sorted[x]
            else:
                if (sizes_sorted[x-1]-sizes_sorted[x])>15:
                    if sizes_sorted[x-1] >= 80:
                        min_size = sizes_sorted[x-1]

        #debug use
        print('sorted size: '+str(sizes_sorted))
        print('min size: '+str(min_size))


        conn_comp2 = np.zeros((output.shape), np.uint8)
        for x in range(0, nb_components):
            if sizes[x] >= min_size:
                conn_comp2[output == x + 1] = 255

        #try to connect the right and left side to draw the whole disc out with 2 iterations of dilate and erode each
        se = np.ones((1,5))
        di_im = cv.dilate(conn_comp2,se, iterations=2)
        er_im = cv.erode(di_im,se, iterations=2)

        #find outter contours of the image and fill in the contours with white
        contours, hierarchy = cv.findContours(er_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        drawing = np.zeros((er_im.shape[0], er_im.shape[1]), dtype=np.uint8)
        for x in range(len(contours)):
            cv.drawContours(drawing, contours, x, (255, 255, 255), cv.FILLED, cv.LINE_8, hierarchy, 0)

        #remove the border drawn before
        final = drawing[10:10+(y1*2),10:10+(x2*2)]

        #for saving images
        im_count += 1 

        #saving locations
        if filePath == 'D:/Users/Alexg/Desktop/MultanT2MRI/Healthy/':
            cv.imwrite('D:/Users/Alexg/Desktop/test/healthy'+str(i)+'_'+str(im_count)+'.jpg', final)

        elif filePath == 'D:/Users/Alexg/Desktop/MultanT2MRI/HERNIATED/':
            cv.imwrite('D:/Users/Alexg/Desktop/test/unhealthy'+str(i)+'_'+str(im_count)+'.jpg', final)

        #for plotting
        row_count = 0

        #plots all images
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(im_ori, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(im_roi, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(cl1, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(res, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(v, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(th, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(bordered, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(conn_comp, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(noise_conn_comp, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(morph_after_noise, cmap='gray')
        row_count +=1      
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(conn_comp2, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(di_im, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(er_im, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(drawing, cmap='gray')
        row_count +=1
        plt.figure(i),plt.subplot(rows,column,count+column*row_count),plt.imshow(final, cmap='gray')


        count += 1

        result_roi.append(im_roi)
        result_segment.append(final)

#for showing original image vs final image
result_roi.insert(0,[])
result_segment.insert(0,[])
counts = 0

if filePath == 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/Healthy/':
    for i in range(1, len(result_roi)):
        if i<56:
            plt.figure(99),plt.subplot(row_res,5,i+(counts*5)),plt.imshow(result_roi[i], cmap='gray')
            plt.figure(99),plt.subplot(row_res,5,i+(counts*5)+5),plt.imshow(result_segment[i], cmap='gray')
        if i%5==0:
            counts +=1
elif filePath == 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/HERNIATED/':
    for i in range(1, len(result_roi)):
        if i<56:
            plt.figure(100),plt.subplot(row_res,5,i+(counts*5)),plt.imshow(result_roi[i], cmap='gray')
            plt.figure(100),plt.subplot(row_res,5,i+(counts*5)+5),plt.imshow(result_segment[i], cmap='gray')

        else:
            plt.figure(101),plt.subplot(row_res,5,i-55+((counts-11)*5)),plt.imshow(result_roi[i], cmap='gray')
            plt.figure(101),plt.subplot(row_res,5,i-55+((counts-11)*5)+5),plt.imshow(result_segment[i], cmap='gray')
        if i%5==0:
            counts +=1
plt.figure(99),plt.suptitle('Images of Healthy Patients Cropped ROI Discs and their Respective Segmentations (Part 1/1)')
plt.figure(100),plt.suptitle('Images of Unhealthy Patients Cropped ROI Discs and their Respective Segmentations (Part 1/2)')
plt.figure(101),plt.suptitle('Images of Unhealthy Patients Cropped ROI Discs and their Respective Segmentations (Part 2/2)')
       
plt.show()
cv.waitKey(0)