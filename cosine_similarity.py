from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import chi2_kernel
import numpy as np
import cv2
import os
def get_data(data_path):

    list1 = []


    # data_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/tsne/input_b/'

    for root, sub_dirs, filelist in os.walk(data_path):
        for filename in filelist:
            list1.append(filename)

    num1 = len(list1)

    label_batch = []


    for i in range (num1):
        if i == 0:
            img = cv2.imread(data_path + list1[i])
            data = np.array(img, dtype=np.single)
            data = np.reshape(data, (1, 512 * 512 * 3))
            data_batch = data
            label_batch.append(1)

        if i != 0:
            img = cv2.imread(data_path + list1[i])
            data = np.array(img, dtype=np.single)
            data = np.reshape(data, (1, 512 * 512 * 3))
            data_batch =  np.concatenate((data_batch,data),axis=0)
            label_batch.append(1)

    label_batch = np.array(label_batch)

    n_samples, n_features = data_batch.shape
    return data_batch, label_batch, n_samples, n_features

if __name__=='__main__':
    data_1, label_1, n_samples_1, n_features_1 = get_data(data_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/Mutil_center/cyc_data/IMG/tsne/1/')
    data_2, label_2, n_samples_2, n_features_2 = get_data(data_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/Mutil_center/cyc_data/IMG/tsne/2/')
    data_3, label_3, n_samples_3, n_features_3 = get_data(data_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/Mutil_center/cyc_data/IMG/tsne/3/')
    data_4, label_4, n_samples_4, n_features_4 = get_data(data_path = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/Mutil_center/cyc_data/IMG/tsne/4/')

    result_1 = cosine_similarity(data_1,data_1)
    result_2 = cosine_similarity(data_1,data_2)
    result_3 = cosine_similarity(data_1,data_3)
    result_4 = cosine_similarity(data_1,data_4)
    result_5 = cosine_similarity(data_2,data_2)
    result_6 = cosine_similarity(data_2,data_3)
    result_7 = cosine_similarity(data_2,data_4)
    result_8 = cosine_similarity(data_3,data_3)
    result_9 = cosine_similarity(data_3,data_4)
    result_10 = cosine_similarity(data_4,data_4)

    print(result_1.mean())
    print(result_2.mean())
    print(result_3.mean())
    print(result_4.mean())
    print(result_5.mean())
    print(result_6.mean())
    print(result_7.mean())
    print(result_8.mean())
    print(result_9.mean())
    print(result_10.mean())


import tensorflow as tf
tf.image.ssim_multiscale
image_1_raw_data = tf.gfile.FastGFile('/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/Mutil_center/drive/data/train/images/21_training.tif','rb').read()
image_2_raw_data = tf.gfile.FastGFile('/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/Mutil_center/drive/data/train/1st_manual/21_manual1.gif','rb').read()
print(image_1_raw_data)

with tf.Session() as sess:
    im1 = tf.image.decode_jpeg(
        image_1_raw_data)
    im2 = tf.image.decode_jpeg(
        image_2_raw_data)
    ssim = tf.image.ssim(im1, im2, max_val=255)
    ssIm=sess.run([ssim])
print(ssIm)