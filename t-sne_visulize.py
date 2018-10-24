
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import cv2
import os


# data = '/home/sdb1/data_analysis/data/drive/'
# cyc_data = '/home/sdb1/data_analysis/data/cyc_drive/'
#
#
# img = cv2.imread(data + 'input_b_4_0.jpg')
#
# img1 = cv2.imread(cyc_data + 'fake_a_4_0.jpg')
#
# data1  = np.array(img, dtype=np.single)
# data2 = np.array(img1, dtype=np.single)





def get_data():

    list1 = []
    list2 = []
    list3 = []
    list4 = []

    data_1 = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/chase/'
    data_2 = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/messdior/'
    data_3 = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/9.27/cyc_img/messdior/'
    # data_4 = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/eye/2/'

    for root, sub_dirs, filelist in os.walk(data_1):
        for filename in filelist:
            list1.append(filename)

    num1 = len(list1)

    for root, sub_dirs, filelist in os.walk(data_2):
        for filename in filelist:
            list2.append(filename)

    num2 = len(list2)


    for root, sub_dirs, filelist in os.walk(data_3):
        for filename in filelist:
            list3.append(filename)

    num3 = len(list3)

    # for root, sub_dirs, filelist in os.walk(data_4):
    #     for filename in filelist:
    #         list4.append(filename)
    #
    # num4 = len(list4)
    #
    label_batch = []


    for i in range (num1):
        if i == 0:
            img = cv2.imread(data_1 + list1[i])
            data = np.array(img, dtype=np.single)
            data = np.resize(data, (1, 512 * 512 * 3))
            data_batch = data
            label_batch.append(1)

        if i != 0:
            img = cv2.imread(data_1 + list1[i])
            data = np.array(img, dtype=np.single)
            data = np.resize(data, (1, 512 * 512 * 3))
            data_batch =  np.concatenate((data_batch,data),axis=0)
            label_batch.append(1)

    for j in range(num2):

        img = cv2.imread(data_2 + list2[j])
        data = np.array(img, dtype=np.single)
        data = np.resize(data, (1, 512 * 512 * 3))
        data_batch = np.concatenate((data_batch, data), axis=0)
        label_batch.append(2)

    for j in range(num3):

        img = cv2.imread(data_3 + list3[j])
        data = np.array(img, dtype=np.single)
        data = np.resize(data, (1, 512 * 512 * 3))
        data_batch = np.concatenate((data_batch, data), axis=0)
        label_batch.append(3)

    # for j in range(num4):
    #
    #     img = cv2.imread(data_4 + list4[j])
    #     data = np.array(img, dtype=np.single)
    #     data = np.resize(data, (1, 512 * 512 * 3))
    #     data_batch = np.concatenate((data_batch, data), axis=0)
    #     label_batch.append(4)

    label_batch = np.array(label_batch)

    n_samples, n_features = data_batch.shape
    return data_batch, label_batch, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min+0.0000000000000000000000000000000001) / (x_max - x_min+0.0000000000000000000000000000000001)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,'t-SNE embedding of the digits (time %.2fs)' % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    main()