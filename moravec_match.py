import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


#这个程序是根据开源moravec实现方案改编的匹配算法
###############calculate moravec###############


def calcV(window1, window2):
    # 用于计算窗口间的灰度差平方和
    win1 = np.int32(window1)
    win2 = np.int32(window2)
    diff = win1 - win2
    diff = diff * diff
    return np.sum(diff)


def getWindow(img, i, j, win_size):
    # 获得指定范围、大小的窗口内容
    if win_size % 2 == 0:
        win = None
        return win
    half_size = win_size / 2

    start_x = round(i - half_size)
    start_y = round(j - half_size)
    end_x = round(i + half_size + 1)-1
    end_y = round(j + half_size + 1)-1
    win = img[start_x:end_x, start_y:end_y]
    return win


def getWindowWithRange(img, i, j, win_size):
    # 获取指定范围、大小的窗口内容以及坐标
    if win_size % 2 == 0:
        win = None
        return win
    half_size = win_size / 2

    start_x = round(i - half_size)
    start_y = round(j - half_size)
    end_x = round(i + half_size + 1)-1
    end_y = round(j + half_size + 1)-1

    win = img[start_x:end_x, start_y:end_y]
    return win, start_x, end_x, start_y, end_y


def get8directionWindow(img, i, j, win_size, win_offset):
    # 获取8个方向的不同窗口内容
    half_size = int(win_size / 2)
    win_tl = img[i - win_offset - half_size:i - win_offset + half_size + 1, j - win_offset - half_size:j - win_offset + half_size + 1]
    win_t = img[i - win_offset - half_size:i - win_offset + half_size + 1, j - half_size:j + half_size + 1]
    win_tr = img[i - win_offset - half_size:i - win_offset + half_size + 1, j + win_offset - half_size:j + win_offset + half_size + 1]
    win_l = img[i - half_size:i + half_size + 1, j - win_offset - half_size:j - win_offset + half_size + 1]
    win_r = img[i - half_size:i + half_size + 1, j + win_offset - half_size:j + win_offset + half_size + 1]
    win_bl = img[i + win_offset - half_size:i + win_offset + half_size + 1, j - win_offset - half_size:j - win_offset + half_size + 1]
    win_b = img[i + win_offset - half_size:i + win_offset + half_size + 1, j - half_size:j + half_size + 1]
    win_br = img[i + win_offset - half_size:i + win_offset + half_size + 1, j + win_offset - half_size:j + win_offset + half_size + 1]
    return win_tl, win_t, win_tr, win_l, win_r, win_bl, win_b, win_br


def nonMaximumSupression(mat, nonMaxValue=0):
    #非极大值抑制
    mask = np.zeros(mat.shape, mat.dtype) + nonMaxValue
    max_value = np.max(mat)#取最大值并定位
    loc = np.where(mat == max_value)
    row = loc[0]
    col = loc[1]
    mask[row, col] = max_value
    return mask, row, col


def getScore(item):
    return item[2]


def getKeypoints(keymap, nonMaxValue, nFeature=-1):
    # 用于获取角点的坐标以及对角点进行排序筛选
    loc = np.where(keymap != nonMaxValue)
    xs = loc[1]
    ys = loc[0]
    print(xs.__len__(), 'keypoints were found.')
    kps = []
    for x, y in zip(xs, ys):
        kps.append([x, y, keymap[y, x]])

    if nFeature != -1:
        kps.sort(key=getScore)
        kps = kps[:nFeature]
        print(kps.__len__(), 'keypoints were selected.')
    return kps


def drawKeypoints(img, kps):    
    for kp in kps:
        pt = (kp[0], kp[1])     
        cv2.circle(img, pt, 3, [0, 0, 255], 1, cv2.LINE_AA)
    return img


def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    #把第一张图片放在左边
    out[:rows1,:cols1] = img1

    #把匹配图片放在右边
    out[:rows2,cols1:] = img2

    #把特征点用线连接
    for mat in matches:
        #获得索引
        img1_idx = mat[0]
        img2_idx = mat[1]

        (x1,y1) = kp1[img1_idx]
        (x2,y2) = kp2[img2_idx]

        if x1==0 or y1==0 or x2==0 or y2==0:
            pass
        else:
            a = np.random.randint(0,256)
            b = np.random.randint(0,256)
            c = np.random.randint(0,256)

            cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (a, b, c), 1)
            cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (a, b, c), 1)

            cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (a, b, c), 1, lineType=cv2.LINE_AA, shift=0)

    return out


def getMoravecKps(img_path, win_size=3, win_offset=1, nonMax_size=5, nonMaxValue=0, nFeature=-1, thCRF=-1):
    """
    将上面的步骤整合为一个函数，方便调用

    :img_path: 影像的路径
    :win_size: 滑动窗口的大小
    :win_offset: 窗口偏移的距离
    :nonMax_size: 非极大值抑制的滑动窗口大小
    :nonMaxValue: 非极大值抑制时，非极大值被赋的值
    :nFeature: 打算提取的角点个数，-1表示自动
    :thCRF: 在对CRF进行筛选时使用的阈值，-1表示自动计算平均值作为阈值

    """
    #优化原moravec四方向的灰度差平方和比较，利用八方向的灰度差平方和做角点判断
    img_rgb = cv2.imread(img_path)
    # print(img_rgb.shape)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_h = img.shape[0]
    img_w = img.shape[1]

    keymap = np.zeros([img_h, img_w], np.int32)

    safe_range = win_offset + win_size#防止超限
    for i in range(safe_range, img_h - safe_range):
        for j in range(safe_range, img_w - safe_range):
            win = getWindow(img, i, j, win_size)
            win_tl, win_t, win_tr, win_l, win_r, win_bl, win_b, win_br = get8directionWindow(img, i, j, win_size, win_offset)
            v1 = calcV(win, win_tl)
            v2 = calcV(win, win_t)
            v3 = calcV(win, win_tr)
            v4 = calcV(win, win_l)
            v5 = calcV(win, win_r)
            v6 = calcV(win, win_bl)
            v7 = calcV(win, win_b)
            v8 = calcV(win, win_br)
            c = min(v1, v2, v3, v4, v5, v6, v7, v8)
            keymap[i, j] = c

    if thCRF == -1:
        # CRF的平均值作为筛选阈值
        mean_c = np.mean(keymap)
    else:
        mean_c = thCRF

    # cv2.imwrite("keymap_test.jpg", keymap)
    keymap = np.where(keymap < mean_c, 0, keymap)
    # cv2.imwrite("keymap_th_test.jpg", keymap)
    
    for i in range(safe_range, img_h - safe_range):
        for j in range(safe_range, img_w - safe_range):
            win, stx, enx, sty, eny = getWindowWithRange(keymap, i, j, nonMax_size)
            nonMax_win, row, col = nonMaximumSupression(win)
            keymap[stx:enx, sty:eny] = nonMax_win
    # cv2.imwrite("keymap_nonMax.jpg", keymap)

    kps = getKeypoints(keymap, nonMaxValue=nonMaxValue, nFeature=nFeature)
    img_kps = drawKeypoints(img_rgb, kps)

    with open('moravec.txt', 'w') as f:
        for kp in kps:
            pt = (kp[0], kp[1])
            f.write("%d %d\n" %(kp[0],kp[1]))

    return kps, img_kps


###############calculate match###############

'''
以下部分为相关匹配系数算法，可以套用文件夹里的my_forstner.py my_harris.py my_moravec.py中提取出的特征点
原理一致
'''



def getMoravec():
    file_name = "moravec.txt"
    f = open(file_name, 'r')
    pts = []
    for lines in f:

        x = int(lines.split(' ')[0])
        y = int(lines.split(' ')[1])

        pts.append([x, y])

    f.close()
    return pts


def getMatch(img_path, match_img_path, x_step=-80, y_step=-20, win_size=15, match_size = 10):
    """
    对每一个特征点和其对应窗口进行匹配计算

    :match_img_path: 影像的路径
    :(x_step, y_step): 右片相对左片的概略相对位移
    :win_size: 滑动窗口大小
    :match_size: 右片固定搜索范围

    """

    img_left_rgb = cv2.imread(img_path)
    img_left = cv2.cvtColor(img_left_rgb, cv2.COLOR_BGR2GRAY)
    img_left_h = img_left.shape[0]
    img_left_w = img_left.shape[1]

    img_right_rgb = cv2.imread(match_img_path)
    img_right = cv2.cvtColor(img_right_rgb, cv2.COLOR_BGR2GRAY)
    img_right_h = img_right.shape[0]
    img_right_w = img_right.shape[1]

    pts = getMoravec()

    count = 0
    match_pts = []
    matches = []
    win_size_2 = int(win_size/2)
    match_size_2 = int(match_size/2)

    for i_this in range(0, int(pts.__len__())):
        start_left_x = pts[i_this][0] - win_size_2
        end_left_x = pts[i_this][0] + win_size_2
        start_left_y = pts[i_this][1] - win_size_2
        end_left_y = pts[i_this][1] + win_size_2

        imax = 0
        jmax = 0

        if start_left_x<0 or end_left_x>img_left_w or start_left_y<0 or end_left_y>img_left_h:
            pass
        else:
            sum_left_G = 0
            sum_left_G2 = 0

            start_x = pts[i_this][0] + x_step - match_size_2 - win_size_2
            end_x = pts[i_this][0] + x_step + match_size_2 + win_size_2
            start_y = pts[i_this][1] + y_step - match_size_2 - win_size_2
            end_y = pts[i_this][1] + y_step + match_size_2 + win_size_2

            start_right_x = pts[i_this][0] + x_step - match_size_2
            end_right_x = pts[i_this][0] + x_step + match_size_2
            start_right_y = pts[i_this][1] + y_step - match_size_2
            end_right_y = pts[i_this][1] + y_step + match_size_2


            if start_x<0 or end_x>img_right_w or start_y<0 or end_y>img_right_h:
                pass
            else:
                for i in range(start_left_x, end_left_x):
                    for j in range(start_left_y, end_left_y):
                        sum_left_G = sum_left_G + img_left[j,i]
                        sum_left_G2 = sum_left_G2 + pow(img_left[j,i], 2)

                for i in range(start_right_x, end_right_x):
                    for j in range(start_right_y, end_right_y):
                        #在搜索框里取范围
                        sum_right_G = 0
                        sum_right_G2 = 0
                        sum_GG = 0
                        for k in range(-win_size_2, win_size_2):
                            for l in range(-win_size_2, win_size_2):
                                sum_right_G = sum_right_G + img_right[j+l,i+k]
                                sum_right_G2 = sum_right_G2 + pow(img_right[j+l,i+k], 2)          
                                GG = int(img_right[j+l,i+k])*int(img_left[start_left_y+win_size_2+k,start_left_x+win_size_2+k])
                                sum_GG = sum_GG + GG

                        relation = 0
                        numer = sum_GG - (sum_left_G*sum_right_G)/(win_size*win_size)
                        deno = (sum_left_G2-pow(sum_left_G/win_size, 2))*(sum_right_G2-pow(sum_right_G/win_size, 2))
                        deno = math.sqrt(deno)
                        try:
                            relationData = numer/deno

                            if i==0 and j==0:
                                relation = relationData
                                imax = i
                                jmax = j
                            else:
                                if relationData>relation:
                                    relation = relationData
                                    imax = i
                                    jmax = j
                        except:
                            pass
        
        match_pts.append([imax, jmax])
        matches.append([i_this,count])


        count = count + 1
    match_img = drawKeypoints(img_right_rgb, match_pts)
    match_imgs = drawMatches(img_left_rgb, pts, img_right_rgb, match_pts, matches)              

    return match_pts, match_img, match_imgs


if __name__ == '__main__':

    print('please input the picture set you want to test: 1 or 2')
    print('recommend 1 because 2 has rotation angles witch the relative matches could not solve')
    index = input()

    pic_path = index+'_left.png'
    match_pic_path = index+'_right.png'

    kps, img = getMoravecKps(pic_path, nFeature=300)
    cv2.imwrite("moravec_test.jpg", img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    #相关系数匹配部分
    #为减小计算量为搜索区域赋予一个初始偏移值，从而利用较小的搜索框获得较好的匹配效果
    if int(index) == 1:
        # x_step = -80
        # y_step = -25
        match_pts, match_img, match_imgs = getMatch(pic_path, match_pic_path, x_step = -80, y_step = -25)
    elif int(index) == 2:
        # x_step = -50
        # y_step = 100
        match_pts, match_img, match_imgs = getMatch(pic_path, match_pic_path, x_step = -50, y_step = 100)
    else:
        print("invalid input")

    
    cv2.imwrite("match_result.jpg", match_img)
    cv2.imwrite("match_align.jpg", match_imgs)