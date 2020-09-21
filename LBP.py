import cv2

import matplotlib.pyplot as plt

import numpy as np



img = cv2.imread('ALL-2views/baby1/view5.png')
imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
org_img = np.transpose(imgYCC, (2, 0, 1))
print(org_img.shape)
pd_img = np.pad(org_img, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=(0, 0))
lbp_img = np.zeros_like(org_img)
r, c = org_img[0].shape
tr, tc = r // 8, c // 8
plt.figure(figsize=(20, 20))
for i in range(3):
    ch_img = pd_img[i]
    for j in range(r):
        for k in range(c):
            t = ch_img[j:j + 3, k:k + 3]
            tt = t >= t[1, 1]
            lbp_img[i][j][k] = np.packbits((tt[1, 0], tt[2, 0], tt[2, 1], tt[2, 2], tt[1, 2], tt[0, 2], tt[0, 1], tt[0, 0]))[0]
    for j in range(7):
        for k in range(7):
            plt.subplot2grid((21, 21), (14 + j, 7 * i + k))
            plt.hist(lbp_img[i, j * tr:(j + 2) * tr, k * tc:(k + 2) * tc], density=1, histtype='stepfilled', facecolor='k')
            plt.xticks([])
            plt.yticks([])

for i in range(3):
    plt.subplot2grid((21, 21), (0, 7 * i), rowspan=7, colspan=7)
    plt.imshow(org_img[i], cmap=plt.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])
    plt.subplot2grid((21, 21), (7, 7 * i), rowspan=7, colspan=7)
    plt.imshow(lbp_img[i], cmap=plt.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])

    cv2.imwrite("output/LBP_right.png", lbp_img[0])
plt.show()