import cv2
import numpy as np
import math

def minu(img1, img2):
    out = img2-img1    
    return out

if __name__ == "__main__":
    img1_path = 'ALL-2views/Baby1/disp1.png'
    img2_path = 'save/left_disparity_map20120.png'

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    out = minu(img1, img2)
    cv2.imwrite("result/20120.jpg", out)
    pass