import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size

#folders
#filePath = 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/Healthy/'
filePath = 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/HERNIATED/'

result_roi = []
result_segment = []
row_res = 22

images = []
for filename in os.listdir(filePath):
    img = cv.imread(os.path.join(filePath,filename))
    if img is not None:
        images.append(img)


for i in range(len(images)):
    im = images[i]
    
    # cv.namedWindow('Selection of ROI', cv.WINDOW_KEEPRATIO)
    # boxes = cv.selectROIs('Selection of ROI', im)

    boxes = []
    count = 1
    rows = 20
    column = 5

    print(i)
    print(boxes)

    if filePath == 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/Healthy/':
        if i == 0:
            boxes = [[ 95, 131,  23,  20],
            [ 86, 114,  26,  21],
            [ 84,  94,  28,  17],
            [ 86,  72,  29,  17],
            [ 93,  54,  24,  13]]

        elif i == 1:
            boxes = [[ 93, 132,  22,  18],
            [ 84, 116,  26,  19],
            [ 85,  97,  25,  15],
            [ 86,  80,  26,  15],
            [ 91,  62,  24,  15]]

        elif i == 2:
            boxes = [[ 73, 129,  21,  15],
            [ 66, 112,  27,  16],
            [ 70,  92,  26,  16],
            [ 77,  72,  24,  16],
            [ 83,  57,  25,  14]]

        elif i == 3:
            boxes = [[ 85, 131,  29,  24],
            [ 80, 112,  29,  19],
            [ 76,  91,  30,  18],
            [ 76,  72,  33,  16],
            [ 82,  53,  28,  14]]

        elif i == 4:
            boxes = [[ 91, 120,  21,  20],
            [ 82, 104,  26,  18],
            [ 82,  87,  27,  13],
            [ 89,  66,  23,  16],
            [ 93,  51,  23,  12]]

        elif i == 5:
            boxes = [[ 91, 171,  21,  15],
            [ 85, 155,  23,  16],
            [ 84, 138,  25,  16],
            [ 87, 120,  24,  15],
            [ 90, 103,  24,  12]]

        elif i == 6:
            boxes = [[ 86, 129,  27,  20],
            [ 82, 110,  29,  16],
            [ 85,  88,  23,  16],
            [ 85,  69,  26,  13],
            [ 88,  50,  23,  13]]

        elif i == 7:
            boxes = [[ 93, 129,  22,  24],
            [ 85, 109,  24,  22],
            [ 81,  90,  25,  19],
            [ 83,  72,  25,  15],
            [ 88,  54,  22,  15]]

        elif i == 8:
            boxes = [[ 82, 125,  24,  20],
            [ 79, 103,  28,  21],
            [ 85,  86,  21,  17],
            [ 86,  68,  25,  14],
            [ 91,  50,  24,  12]]

        elif i == 9:
            boxes = [[ 82, 125,  25,  22],
            [ 79, 108,  29,  16],
            [ 81,  87,  28,  16],
            [ 86,  68,  23,  15],
            [ 91,  51,  23,  13]]

        else:
            print("ERROR AT TOP")

    elif filePath == 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/HERNIATED/':
        if i==0:
            boxes = [[ 81, 128,  23,  20],
            [ 73, 110,  27,  15],
            [ 75,  90,  25,  13],
            [ 78,  70,  26,  12],
            [ 83,  50,  25,  12]]

        elif i==1:
            boxes=[[ 71, 133,  33,  28],
            [ 66, 113,  30,  24],
            [ 64,  90,  32,  21],
            [ 65,  67,  30,  18],
            [ 69,  46,  29,  18]]

        elif i==2:
            boxes=[[ 80, 146,  26,  18],
            [ 72, 125,  36,  22],
            [ 73, 107,  32,  17],
            [ 80,  86,  27,  16],
            [ 83,  70,  28,  13]]
         
        elif i==3:
            boxes=[[ 83, 131,  27,  22],
            [ 80, 115,  28,  17],
            [ 83,  97,  28,  12],
            [ 87,  77,  31,  16],
            [ 93,  60,  27,  17]]
        
        elif i==4:
            boxes=[[ 85, 129,  30,  20],
            [ 85, 111,  25,  19],
            [ 83,  90,  27,  17],
            [ 86,  71,  26,  15],
            [ 87,  53,  28,  15]]

        elif i==5:
            boxes=[[ 72, 132,  30,  22],
            [ 69, 110,  30,  20],
            [ 69,  87,  32,  19],
            [ 74,  71,  30,  15],
            [ 78,  48,  28,  16]]

        elif i==6:
            boxes=[[ 87, 135,  23,  21],
            [ 80, 118,  28,  16],
            [ 83, 101,  24,  14],
            [ 86,  82,  26,  15],
            [ 88,  63,  25,  13]]

        elif i==7:
            boxes=[[ 90, 132,  30,  20],
            [ 84, 114,  33,  20],
            [ 88,  92,  28,  17],
            [ 88,  72,  32,  16],
            [ 95,  53,  26,  16]]

        elif i==8:
            boxes=[[ 78, 134,  25,  23],
            [ 70, 118,  29,  20],
            [ 67, 101,  29,  18],
            [ 70,  85,  27,  14],
            [ 77,  65,  24,  16]]


        elif i==9:
            boxes=[[ 74, 132,  26,  27],
            [ 64, 117,  34,  19],
            [ 65,  96,  31,  13],
            [ 68,  73,  29,  16],
            [ 73,  52,  27,  16]]

        elif i==10:
            boxes=[[ 80, 139,  24,  23],
            [ 69, 120,  30,  25],
            [ 66, 103,  36,  18],
            [ 71,  84,  31,  14],
            [ 74,  64,  29,  15]]

        elif i==11:
            boxes=[[102, 128,  23,  21],
            [ 89, 112,  32,  26],
            [ 83,  95,  29,  22],
            [ 81,  77,  28,  17],
            [ 84,  57,  27,  17]]

        elif i==12:
            boxes=[[ 90, 127,  30,  21],
            [ 86, 108,  29,  21],
            [ 85,  86,  27,  20],
            [ 81,  69,  30,  15],
            [ 81,  50,  29,  14]]

        elif i==13:
            boxes=[[ 82, 129,  29,  21],
            [ 78, 113,  33,  14],
            [ 79,  89,  33,  17],
            [ 80,  69,  33,  18],
            [ 84,  52,  27,  13]]

        elif i==14:
            boxes=[[ 89, 141,  25,  22],
            [ 82, 122,  29,  20],
            [ 80, 106,  26,  17],
            [ 78,  87,  29,  16],
            [ 83,  66,  26,  17]]

        elif i==15:
            boxes=[[ 77, 133,  27,  21],
            [ 71, 112,  34,  19],
            [ 73,  93,  32,  15],
            [ 79,  71,  30,  16],
            [ 82,  52,  30,  16]]

        elif i==16:
            boxes=[[ 86, 147,  26,  21],
            [ 80, 130,  30,  19],
            [ 77, 115,  30,  16],
            [ 77,  98,  28,  14],
            [ 80,  79,  28,  13]]

        elif i==17:
            boxes=[[ 86, 126,  26,  24],
            [ 82, 109,  26,  20],
            [ 80,  90,  26,  16],
            [ 81,  68,  28,  17],
            [ 83,  50,  26,  14]]

        elif i==18:
            boxes=[[ 73, 128,  26,  21],
            [ 65, 110,  29,  19],
            [ 63,  91,  28,  16],
            [ 64,  70,  29,  16],
            [ 67,  51,  26,  12]]

        elif i==19:
            boxes=[[ 75, 134,  31,  23],
            [ 71, 111,  31,  24],
            [ 71,  93,  32,  19],
            [ 73,  73,  31,  16],
            [ 77,  52,  29,  14]]

        elif i==20:
            boxes=[[101, 134,  28,  18],
            [ 98, 117,  31,  17],
            [ 96,  99,  34,  17],
            [ 98,  75,  30,  21],
            [ 98,  60,  27,  17]]

        elif i==21:
            boxes=[[ 93, 130,  32,  23],
            [ 84, 115,  35,  20],
            [ 84,  96,  33,  16],
            [ 81,  74,  38,  17],
            [ 83,  55,  34,  17]]
        
        else:
            print("Error occured")

    else:
        print("ERROR GETTING FILE")

    im_count = 0

    for rect in boxes:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        im_ori = im[y1:y1+y2,x1:x1+x2]
        #im_roi = cv.resize(im_roi, dsize=None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
        im_roi = cv.resize(im_ori, dsize=None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        im_roi = cv.cvtColor(im_roi, cv.COLOR_BGR2GRAY)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(im_roi)

        lookUpTable = np.empty((1,256), np.uint8)
        for x in range(256):
            lookUpTable[0,x] = np.clip(pow(x / 255.0, 0.8) * 255.0, 0, 255)
        res = cv.LUT(cl1, lookUpTable)

        #normal thresholding using range
        #for the black ring around each disc
        temp = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        temp = cv.cvtColor(temp,cv.COLOR_BGR2HSV)

        h,s,v = cv.split(temp)

        lowVal = 0#0/100*255 
        highVal = 120#50/100*255 #47.5
        roi = cv.inRange(v, lowVal, highVal)
        ret,th = cv.threshold(roi,0,255,cv.THRESH_BINARY)

        # th_adap = cv.adaptiveThreshold(v, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 0)
        # ret,th_otsu = cv.threshold(v, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        bordered = cv.copyMakeBorder(th, 10,10,10,10,cv.BORDER_CONSTANT, 0)
        
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(bordered, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        #calc largest 2 component
        sizes_sorted = sorted(sizes, reverse=True)
        sizes_sorted.append(0)
        
        for x in range(0, len(sizes_sorted)):
            if len(sizes_sorted) == 1:
                print('no comparison, only 1 data')
                min_size = sizes_sorted[x]
            else:
                if (sizes_sorted[x-1]-sizes_sorted[x])>15:
                    if sizes_sorted[x-1] >= 80:
                        min_size = sizes_sorted[x-1]

        #your answer image
        conn_comp = np.zeros((output.shape), np.uint8)
        #for every component in the image, you keep it only if it's above min_size
        for x in range(0, nb_components):
            if sizes[x] >= min_size:
                conn_comp[output == x + 1] = 255

        noise_conn_comp = cv.medianBlur(conn_comp,3)
        se = np.ones((1,3))
        morph_after_noise = cv.erode(noise_conn_comp,se)
        morph_after_noise = cv.dilate(morph_after_noise,se)

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

        print('sorted size: '+str(sizes_sorted))
        print('min size: '+str(min_size))

        conn_comp2 = np.zeros((output.shape), np.uint8)
        #for every component in the image, you keep it only if it's above min_size
        for x in range(0, nb_components):
            if sizes[x] >= min_size:
                conn_comp2[output == x + 1] = 255

        se = np.ones((1,5))
        di_im = cv.dilate(conn_comp2,se, iterations=2)
        er_im = cv.erode(di_im,se, iterations=2)

        contours, hierarchy = cv.findContours(er_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        drawing = np.zeros((er_im.shape[0], er_im.shape[1]), dtype=np.uint8)
        for x in range(len(contours)):
            cv.drawContours(drawing, contours, x, (255, 255, 255), cv.FILLED, cv.LINE_8, hierarchy, 0)

        final = drawing[10:10+(y2*2),10:10+(x2*2)]

        im_count += 1 

        if filePath == 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/Healthy/':
            cv.imwrite('D:/Users/Alexg/Desktop/Im Pro Ass/test/healthy'+str(i)+'_'+str(im_count)+'.jpg', final)

        elif filePath == 'D:/Users/Alexg/Desktop/Im Pro Ass/MultanT2MRI/HERNIATED/':
            cv.imwrite('D:/Users/Alexg/Desktop/Im Pro Ass/test/unhealthy'+str(i)+'_'+str(im_count)+'.jpg', final)

        row_count = 0

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
print('No error')