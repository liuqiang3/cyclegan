import tensorflow as tf

def read_and_decode(filename,img_width,img_height):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'train/image': tf.FixedLenFeature([], tf.string), })
    img = tf.decode_raw(features['train/image'], tf.float32)
    img = tf.reshape(img, [512, 512, 3])
    img = tf.image.resize_images(img,(img_width,img_height))
    img=tf.cast(img, tf.float32)
    img = tf.subtract(tf.multiply(tf.div(img, 255),2), 1)



    return img

def createBatch(filename, batchsize,img_width,img_height):
    images = read_and_decode(filename,img_width,img_height)

    min_after_dequeue = 1
    capacity = 10

    image_batch = tf.train.shuffle_batch([images],
                                                      batch_size=batchsize,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue,
                                                      )

    return image_batch
