import cv2
import numpy as np
import math

def combo(img1, img2, img3, img4):
    r1 = img1.shape[0]
    c1 = img1.shape[1]
    r2 = img2.shape[0]
    c2 = img2.shape[1]
    r3 = img3.shape[0]
    c3 = img3.shape[1]
    r4 = img4.shape[0]
    c4 = img4.shape[1]

    rows = max([r1+r3, r2+r4])
    cols = max([c1+c2, c3+c4])
    out = np.zeros((rows, cols, 3), dtype='uint8')

    #把第一张图片放在左上
    out[:r1,:c1] = img1

    #把第二张图片放在右上
    out[:r2,c1:] = img2

    #把第三张图片放在左下
    out[r1:,:c3] = img3

    #把第四张图片放在右下
    out[r2:,c3:] = img4
    
    return out

if __name__ == "__main__":
    img1_path = 'ALL-2views/Baby1/view1.png'
    img2_path = 'ALL-2views/Baby1/view5.png'
    output_img1_path = 'output/left_disparity_map.png'
    output_img2_path = 'output/right_disparity_map.png'

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    output_img1 = cv2.imread(output_img1_path)
    output_img2 = cv2.imread(output_img2_path)
    
    out = combo(img1, img2, output_img1, output_img2)
    cv2.imwrite("result/img_baby_P1_20_P2_70.jpg", out)
    pass