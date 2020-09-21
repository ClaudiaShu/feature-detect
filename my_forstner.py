import numpy as np
import cv2
import config

def get_robert():
    pass


def cal_forstner(win_size=5, win_offset=20):
    img = cv2.imread(config.img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    row = img_gray.shape[0]#801
    col = img_gray.shape[1]#521
    min_val = np.zeros(img_gray.shape)

    pts = []

    
    # range前闭后开
    for r in range(int(win_size/2),row-int(win_size/2)):   
        for c in range(int(win_size/2),col-int(win_size/2)):
            Gu2 = 0
            Gv2 = 0
            GuGv = 0
            for i in range(-int(win_size/2),int(win_size/2)):
                gu = img_gray[r+i+1,c+i+1]-img_gray[r+i,c+i]
                gv = img_gray[r+i,c+i+1]-img_gray[r+i,c]
                Gu2 = Gu2 + pow(gu,2)
                Gv2 = Gv2 + pow(gv,2)
                GuGv = GuGv + gu*gv
            N = np.array([[Gu2,GuGv],[GuGv,Gv2]])
            # Q = np.linalg.inv(N)
            w = np.linalg.det(N)/np.trace(N)
            q = 4*np.linalg.det(N)/pow(np.trace(N),2)#圆度
            # print(q)

            if q>100:
                pts.append([c,r])
                min_val[r,c] = img_gray[r,c]            

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

    cv2.imwrite('Detected_Forstner.jpg', img) 

    return img


if __name__ == "__main__":
    img = cal_forstner()
    cv2.imwrite('Detected_Forstner.jpg', img)
    pass