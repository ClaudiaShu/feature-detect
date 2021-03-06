import cv2
import numpy as np 
import math
import scipy.stats as st
import scipy

output_dir = 'output/'
SIFT_ORI_SIG_FCTR = 1.52
SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
SIFT_ORI_PEAK_RATIO = 0.8
SIFT_INT_DESCR_FCTR = 512.0
# SIFT_FIXPT_SCALE = 48
SIFT_FIXPT_SCALE = 1
SIFT_DESCR_WIDTH=4
SIFT_DESCR_HIST_BINS=8
SIFT_DESCR_SCL_FCTR=3.0
SIFT_DESCR_MAG_THR=0.2
FLT_EPSILON=1.19209290E-07


def scalingandblurringoutput(image,xscale,yscale,kernlen,nsig):
    scalingimage = cv2.resize(image, (0,0), fx=xscale, fy=yscale) 
    rows,columns= scalingimage.shape
    # print(rows,columns)
    gauss_blur_filter = [[0 for x in range(5)] for y in range(5)]
    dummy_image_x = [[0 for x in range(columns)] for y in range(rows)]
    new_image = [[0 for x in range(columns + 6)] for y in range(rows + 6 )]
    pad_image = np.asarray(scalingimage)
    for i in range(rows+3):
        for j in range(columns+3):
            if i == 0 or i== 1 or i == rows+1 or i == rows+2 or i == rows+3:
                new_image[i][j] = 0
            elif j == 0 or j == 1 or j == columns+1 or j == columns+2 or j == columns+3:
                new_image[i][j] = 0
            else:
                new_image[i][j] = pad_image[i-3][j-3]    
    gauss_blur_filter = np.asarray(gaussian_blur_matrix(kernlen,nsig))

    for i in range(rows):
        for j in range(columns):
            dummy_image_x[i][j] = new_image[i][j] * gauss_blur_filter[0][0] + new_image[i][j+1] * gauss_blur_filter[0][1] + new_image[i][j+2] * gauss_blur_filter[0][2] + new_image[i][j+3] * gauss_blur_filter[0][3] +new_image[i][j+4] * gauss_blur_filter[0][4] + new_image[i][j+5] * gauss_blur_filter[0][5] + new_image[i][j+6] * gauss_blur_filter[0][6] + new_image[i+1][j] * gauss_blur_filter[1][0] + new_image[i+1][j+1] * gauss_blur_filter[1][1] + new_image[i+1][j+2] * gauss_blur_filter[1][2] + new_image[i+1][j+3] * gauss_blur_filter[1][3] + new_image[i+1][j+4] * gauss_blur_filter[1][4] + new_image[i+1][j+5] * gauss_blur_filter[1][5] + new_image[i+1][j+6] * gauss_blur_filter[1][6]+ new_image[i+2][j] * gauss_blur_filter[2][0] + new_image[i+2][j+1] * gauss_blur_filter[2][1] + new_image[i+2][j+2] * gauss_blur_filter[2][2] + new_image[i+2][j+3] * gauss_blur_filter[2][3] + new_image[i+2][j+4] * gauss_blur_filter[2][4] +new_image[i+2][j+5] * gauss_blur_filter[2][5] + new_image[i+2][j+6] * gauss_blur_filter[2][6] +new_image[i+3][j] * gauss_blur_filter[3][0] + new_image[i+3][j+1] * gauss_blur_filter[3][1] +new_image[i+3][j+2] * gauss_blur_filter[3][2] + new_image[i+3][j+3] * gauss_blur_filter[3][3] + new_image[i+3][j+4] * gauss_blur_filter[3][4] + new_image[i+3][j+5] * gauss_blur_filter[3][5] + new_image[i+3][j+6] * gauss_blur_filter[3][6]+ new_image[i+4][j] * gauss_blur_filter[4][0] + new_image[i+4][j+1] * gauss_blur_filter[4][1] + new_image[i+4][j+2] * gauss_blur_filter[4][2] + new_image[i+4][j+3] * gauss_blur_filter[4][3] + new_image[i+4][j+4] * gauss_blur_filter[4][4] + new_image[i+4][j+5] * gauss_blur_filter[4][5] + new_image[i+4][j+6] * gauss_blur_filter[4][6] + new_image[i+5][j] * gauss_blur_filter[5][0] + new_image[i+5][j+1] * gauss_blur_filter[5][1] + new_image[i+5][j+2] * gauss_blur_filter[5][2] + new_image[i+5][j+3] * gauss_blur_filter[5][3] + new_image[i+5][j+4] * gauss_blur_filter[5][4] + new_image[i+5][j+5] * gauss_blur_filter[5][5] + new_image[i+5][j+6] * gauss_blur_filter[5][6] + new_image[i+6][j] * gauss_blur_filter[6][0] + new_image[i+6][j+1] * gauss_blur_filter[6][1] + new_image[i+6][j+2] * gauss_blur_filter[6][2] + new_image[i+6][j+3] * gauss_blur_filter[6][3] + new_image[i+6][j+4] * gauss_blur_filter[6][4] + new_image[i+6][j+5] * gauss_blur_filter[6][5] + new_image[i+6][j+6] * gauss_blur_filter[6][6]
    maximum = 0
    for i in range(rows):
        for j in range(columns):
            if maximum < dummy_image_x[i][j] :
                maximum = dummy_image_x[i][j]
           
    minimum = 0

    for i in range(rows):
        for j in range(columns):
            val = dummy_image_x[i][j]
            constant =(val - minimum) / (maximum - minimum)
            dummy_image_x[i][j] = constant
    return (np.asarray(dummy_image_x))

def gaussian_blur_matrix(kernlen, nsig):    
    #kernlen = 7
    #nsig = 0.707
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def difference_of_images(image1,image2):
    #影像差值    
    rows,columns = image1.shape
    resultant_image = [[0 for x in range(columns)] for y in range(rows)]

    for i in range(rows):
        for j in range(columns):
            resultant_image[i][j] = image1[i][j] - image2[i][j]
    return np.asarray(resultant_image)        

def padding(image):
    #扩充
    rows,columns = len(image),len(image[0])
    pad_image = np.asarray(image) 
    new_image = [[0 for x in range(columns + 2)] for y in range(rows + 2 )]
    for i in range(rows+1):
        for j in range(columns+1):
            if i == 0 or i == rows+1:
                new_image[i][j] = 0
            elif j == 0 or j == columns+1:
                new_image[i][j] = 0
            else:
                new_image[i][j] = pad_image[i-1][j-1]
    return new_image

def get_main_direction(img, r, c, radius, sigma, scale, BinNum=36):
    expf_scale = scale

    X = []
    Y = []
    W = []
    temphist = []
    for i in range(BinNum):
        temphist.append(0.0)
    # 图像梯度直方图统计的像素范围
    k = 0
    for i in range(-radius, radius + 1):
        y = r + i
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius, radius + 1):
            x = c + j
            if x <= 0 or x >= img.shape[1] - 1:
                continue
            dx = (img[y, x + 1] - img[y, x - 1])
            dy = (img[y - 1, x] - img[y + 1, x])
            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1

    length = k

    W = np.exp(np.array(W))
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5

    # 计算直方图的每个bin
    for k in range(length):
        bin = int(np.round((BinNum / 360.0) * Ori[k]))
        if bin >= BinNum:
            bin -= BinNum
        if bin < 0:
            bin += BinNum
        temphist[bin] += W[k] * Mag[k]

    # smooth the histogram
    # 高斯平滑
    temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]
    temphist.insert(0, temp[0])
    temphist.insert(0, temp[1])
    temphist.insert(len(temphist), temp[2])
    temphist.insert(len(temphist), temp[3])  # padding

    hist = []
    for i in range(BinNum):
        hist.append(
            (temphist[i] + temphist[i + 4]) * (1.0 / 16.0) + (temphist[i + 1] + temphist[i + 3]) * (4.0 / 16.0) +
            temphist[i + 2] * (6.0 / 16.0))

    # 得到主方向
    maxval = max(hist)
    print(maxval)

    return maxval, hist

def key_points_detection_algo(scale, sigma, image1,image2,image3):
    #空间极值探测
    imagefirst = np.asarray(image1)
    rows,columns = imagefirst.shape
    image1_new = padding(image1)
    image2_new = padding(image2)
    image3_new = padding(image3)
    key_points_detect = [[0 for x in range(columns)] for y in range(rows)]
    kps = []
    OMAX = []
    HIST = []
    #在二维图像空间，中心点与它3*3邻域内的8个点做比较，在同一组内的尺度空间上，中心点和上下相邻的两层图像的2*9个点作比较  
    for i in range(rows):
            for j in range(columns):
                value = image2[i][j]
                max_value = max(image1_new[i][j],image1_new[i][j+1],image1_new[i][j+2],\
                    image1_new[i+1][j],image1_new[i+1][j+1],image1_new[i+1][j+2],\
                        image1_new[i+2][j],image1_new[i+2][j+1],image1_new[i+2][j+2],\
                            image2_new[i][j],image2_new[i][j+1],image2_new[i][j+2],\
                                image2_new[i+1][j],image2_new[i+1][j+1],image2_new[i+1][j+2],\
                                    image2_new[i+2][j],image2_new[i+2][j+1],image2_new[i+2][j+2],\
                                        image3_new[i][j],image3_new[i][j+1],image3_new[i][j+2],\
                                            image3_new[i+1][j],image3_new[i+1][j+1],image3_new[i+1][j+2],\
                                                image3_new[i+2][j],image3_new[i+2][j+1],image3_new[i+2][j+2])    
                min_value = min(image1_new[i][j],image1_new[i][j+1],image1_new[i][j+2],\
                    image1_new[i+1][j],image1_new[i+1][j+1],image1_new[i+1][j+2],\
                        image1_new[i+2][j],image1_new[i+2][j+1],image1_new[i+2][j+2],\
                            image2_new[i][j],image2_new[i][j+1],image2_new[i][j+2],\
                                image2_new[i+1][j],image2_new[i+1][j+1],image2_new[i+1][j+2],\
                                    image2_new[i+2][j],image2_new[i+2][j+1],image2_new[i+2][j+2],\
                                        image3_new[i][j],image3_new[i][j+1],image3_new[i][j+2],\
                                            image3_new[i+1][j],image3_new[i+1][j+1],image3_new[i+1][j+2],\
                                                image3_new[i+2][j],image3_new[i+2][j+1],image3_new[i+2][j+2])
                #写入结果矩阵
                if value == max_value or value == min_value:
                    #提取特征点
                    key_points_detect[i][j] = 255
                    kps.append([i,j])

                    #增加计算主方向函数，并返回特征点主方向描述
                    #omax, hist = get_main_direction(img, r, c, radius, sigma, BinNum)
                    sg = sigma
                    sc = scale
                    omax, hist = get_main_direction(image1, i, j, radius=10, sigma = sg, scale = sc)
                    # print(omax, hist)
                    OMAX.append(omax)
                    HIST.append(hist)

                else:
                    key_points_detect[i][j] = 0
    

    return np.asarray(key_points_detect), kps, OMAX, HIST


def calcDescriptors(gpyr, keypoints):
    return 0

def detect_feature(img_name):
    print("process the left image")
    image = cv2.imread(img_name,0)
    image_rgb = cv2.imread(img_name)
    scaling = [1,1/2,1/4,1/8]
    sigma_values = [[0 for x in range(5)] for y in range(4)]

    constant = 1  

    for i in range(4):
        for j in range(5):
            if i == 0 and j == 0:
                sigma_values[i][j] = 1 / math.sqrt(2)
                constant = sigma_values[i][j]
                continue
            sigma_values[i][j] = constant * math.sqrt(2)
            constant = sigma_values[i][j]
        constant = sigma_values[i][1]
        #print(constant)
    #print(sigma_values)

    new_image = [[0 for x in range(5)] for y in range(4)]
    for i in range(4):
        for j in range(5):
            print("scaling value==================>"+str(scaling[i]))
            print("sigma value==================>"+str(sigma_values[i][j]))
            new_image[i][j] = scalingandblurringoutput(image,scaling[i],scaling[i],7,sigma_values[i][j])
            # output_string = output_dir + "Gaussblur" + str(i) + str(j)+".jpg"
            # cv2.imwrite(output_string , new_image[i][j]*255)
    difference_of_gaussians_image  = [[0 for x in range(4)] for y in range(4)]

    print('calculating DOG')
    for i in range(4):
        for j in range(4):
            #差分高斯计算（相邻高斯卷积图像差值）            
            difference_of_gaussians_image[i][j] = difference_of_images(new_image[i][j],new_image[i][j+1])
            # output_string = output_dir + "DOG" + str(i) + str(j)+".jpg"
            # cv2.imwrite(output_string , difference_of_gaussians_image[i][j]*255)   
             
    print('detection in the DOG space')
    key_points_detection = [[0 for x in range(2)] for y in range(4)]
    kps = [[0 for x in range(2)] for y in range(4)]
    omax = [[0 for x in range(2)] for y in range(4)]
    hist = [[0 for x in range(2)] for y in range(4)]
    
    for i in range(4):
        for j in range(2):
            #在DOG尺度空间中检测(添加方向计算)
            scale = scaling[i]
            sigma = sigma_values[i][j]
            key_points_detection[i][j], kps[i][j], omax[i][j], hist[i][j] = key_points_detection_algo(scale, sigma, difference_of_gaussians_image[i][j],difference_of_gaussians_image[i][j+1],difference_of_gaussians_image[i][j+2])
            
            #SIFT描述子计算
            # desc = calcDescriptors(difference_of_gaussians_image[i][j+1], kps[i,j])

            # output_string = output_dir + "Keypoints" + str(i) + str(j)+".jpg"
            # cv2.imwrite(output_string , key_points_detection[i][j])


    # cv2.imwrite(output_dir + 'Final_image.jpg', image)  
    # cv2.imwrite('sift.jpg', np.asarray(key_points_detection[0][0]))
    print('draw feature pts')
    for pt in kps[0][0]:
        fpt = (pt[1],pt[0])
        cv2.circle(image_rgb, fpt, 3, [0, 0, 255], 1, cv2.LINE_AA)

    cv2.imwrite('sift_'+img_name, image_rgb)
    print(kps[0][0])


     
    # cv2.imshow('task2',np.asarray(key_points_detection[0][0]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return kps[0][0]




if __name__ == "__main__":
    kp1 = detect_feature('1_left.png')
    # kp2 = detect_feature('1_right.png')
    pass