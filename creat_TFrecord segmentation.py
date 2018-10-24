from random import shuffle
import numpy as np
import glob
import tensorflow as tf
import cv2
import sys
import os


shuffle_data = True
image_path_1_1 = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/cycle chase/all_image/*.png'
image_path_2_1 = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/cycle chase/all_label/*.png'
# 取得该路径下所有图片的路径，type（addrs）= list
addrs_1_1 = sorted(glob.glob(image_path_1_1))
addrs_2_1 = sorted(glob.glob(image_path_2_1))
addrs = addrs_1_1
print(addrs)

# 标签数据的获得具体情况具体分析，type（labels）= list
labels = addrs_2_1
print(labels)
# 这里是打乱数据的顺序
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# 按需分割数据集
train_addrs = addrs[0:int(1 * len(addrs))]
train_labels = labels[0:int(1 * len(labels))]

#val_addrs = addrs[int(0.7 * len(addrs)):int(0.9 * len(addrs))]
#val_labels = labels[int(0.7 * len(labels)):int(0.9 * len(labels))]

#test_addrs = addrs[int(0.9 * len(addrs)):]
#test_labels = labels[int(0.9 * len(labels)):]
img_wide = 512
img_height = 512
# 上面不是获得了image的地址么，下面这个函数就是根据地址获取图片
def load_image_3_channel(addr):  # A function to Load image
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_wide, img_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 这里/255是为了将像素值归一化到[0，1]
    #img = img / 255.
    img = img.astype(np.float32)
    return img
def load_image_1_channel(addr):  # A function to Load image
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_wide, img_wide), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 这里/255是为了将像素值归一化到[0，1]
    ret, img = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
    img = img.astype(np.float32)
    return img


# 将数据转化成对应的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# 下面这段就开始把数据写入TFRecods文件

train_filename = '/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/cycle chase/tfrecord/chase_cyc.tfrecords'  # 输出文件地址

# 创建一个writer来写 TFRecords 文件
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
    # 这是写入操作可视化处理
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
        # 加载图片
    img = load_image_3_channel(train_addrs[i])
    print(img.shape)
    label = load_image_1_channel(train_labels[i])
    print(label.shape)

    # 创建一个属性（feature）
    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    # 创建一个 example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # 将上面的example protocol buffer写入文件
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()
label_name=train_filename + 'note'
txtfile = open(label_name, 'w+')
txtfile.writelines('image:'+str(addrs) + "\n"+
                    'label:'+str(labels) + "\n")

