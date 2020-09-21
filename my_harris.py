import cv2

import numpy as np

def cal_harris():

    im = cv2.imread("1_left.png") 
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    R = np.zeros(im_gray.shape) 

    im_gray = cv2.GaussianBlur(im_gray,(5,5),1)

    Ix = cv2.Sobel(im_gray, cv2.CV_64F, 1, 0, ksize=5) 
    Iy = cv2.Sobel(im_gray, cv2.CV_64F, 0, 1, ksize=5)

    A = cv2.GaussianBlur(np.multiply(Ix, Ix),(5,5),1) 
    B = cv2.GaussianBlur(np.multiply(Iy, Iy),(5,5),1)  
    C = cv2.GaussianBlur(np.multiply(Ix, Iy),(5,5),1)   

    k = 0.04
    R = np.multiply(A,B)-np.square(C)-k*np.square(A+B)
    #det(M)-ktr^2(M) k = 0.04

    max_Val = 0.01*R.max()
    for row_number,row in enumerate(R):
        for col_number,val in enumerate(row):
            if val > max_Val:  # Corner Detector
                im[row_number,col_number] = [0,0,255] 
            if val < -max_Val:  # Edge Detector
                im[row_number,col_number] = [0,255,0] 
            else: 
                pass

    cv2.imwrite("Detected_Harris.jpg", im)

    return im
            

if __name__ == "__main__":

    cal_harris()