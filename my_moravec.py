import numpy as np
import cv2
import config


def cal_moravec(win_size=5, win_offset=5,scale=1000):
    img = cv2.imread(config.img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    row = img_gray.shape[0]#801
    col = img_gray.shape[1]#521

    min_val = np.zeros(img_gray.shape)
    threshold = 180
    pts = []

    
    # range前闭后开
    for r in range(int(win_size/2),row-int(win_size/2)):   
        for c in range(int(win_size/2),col-int(win_size/2)):
            V1 = 0
            V2 = 0
            V3 = 0
            V4 = 0
            for i in range(-int(win_size/2),int(win_size/2)):
                V1 = V1+pow((img_gray[r,c+i]-img_gray[r,c+i+1]),2)/scale
                V2 = V2+pow((img_gray[r+i,c+i]-img_gray[r+i+1,c+i+1]),2)/scale
                V3 = V3+pow((img_gray[r+i,c]-img_gray[r+i+1,c]),2)/scale
                V4 = V4+pow((img_gray[r-i,c+i]-img_gray[r-i-1,c+i+1]),2)/scale

                # print(V1)

            min_val[r,c] = min(V1,V2,V3,V4)
            
            # print(min_val[r,c])
            if min_val[r,c]<threshold:
                min_val[r,c] = 0

            else:
                pass
    

    for r in range(win_offset,row-win_offset):   
        for c in range(win_offset,col-win_offset):
            mat = min_val[r-win_offset:r+win_offset,c-win_offset:c+win_offset]
            if np.max(mat)==0:
                pass
            else:
                pos = np.unravel_index(np.argmax(mat),mat.shape)
                R = r+pos[0]-win_offset
                C = c+pos[1]-win_offset
                # print(R,C)
                # feature_map[R,C] = img_gray[R,C]                
                cv2.circle(img, (C,R), 3, [0, 0, 255], 1, cv2.LINE_AA)

    return img

if __name__ == "__main__":
    img = cal_moravec()
    cv2.imwrite('Detected_Moravec.jpg', img)
    pass