from skimage.measure import compare_psnr as PSNR
from skimage.measure import compare_ssim as MSSIM
import os
import cv2

#均值哈希算法
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#差值感知算法
def dHash(img):
    #缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n





list_1 = []
list_2 = []
data_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/1/'
gt_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/2/'
for root, sub_dirs, filelist in os.walk(data_path):
    for filename in filelist:
        list_1.append(filename)
for root, sub_dirs, filelist in os.walk(gt_path):
    for filename in filelist:
        list_2.append(filename)
num = len(list_1)
list_1 = sorted(list_1)
list_2 = sorted(list_2)

result_all = 0
for i in range(num):
    img = cv2.imread(data_path + list_1[i])
    gt = cv2.imread(gt_path + list_2[i])
    # hash1 = aHash(img)
    # hash2 = aHash(gt)
    result = MSSIM(img,gt,multichannel=True)
    result_all = result_all +result
result_mean = result_all / num
print(result_mean)