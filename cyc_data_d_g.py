import tensorflow as tf
import random
import time
import os
import numpy as np
from scipy.misc import imsave


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def read_and_decode(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'train/label': tf.FixedLenFeature([], tf.string),
                  'train/image': tf.FixedLenFeature([], tf.string), })
    img = tf.decode_raw(features['train/image'], tf.float32)
    img = tf.reshape(img, [512, 512, 3])
    img=tf.cast(img, tf.float32)
    img = tf.subtract(tf.multiply(tf.div(img, 255),2), 1)

    #img = tf.subtract(tf.multiply(tf.div(img, 255),2), 1)
    labels = tf.decode_raw(features['train/label'], tf.float32)
    labels = tf.reshape(labels,[512,512,1])
    labels=tf.cast(labels, tf.float32)

    return img, labels

def createBatch(filename, batchsize):
    images, labels = read_and_decode(filename)

    min_after_dequeue = 1
    capacity = 10

    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                      batch_size=batchsize,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue,
                                                      )

    return image_batch, label_batch






batch_size = 1
pool_size = 50
ngf = 32
ndf = 64

img_height = 512
img_width = 512
img_layer = 3
img_size = img_height * img_width
input_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer], name="input_A")
input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer], name="input_B")
gt_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 1], name="gt_A")
gt_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 1], name="gt_B")

fake_pool_A = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="fake_pool_A")
fake_pool_B = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="fake_pool_B")

global_step = tf.Variable(0, name="global_step", trainable=False)
num_fake_inputs = 0
lr = tf.placeholder(tf.float32, shape=[], name="lr")
fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))
A_IMG=[]
B_IMG=[]

def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)


def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
    return out


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d",
                   do_norm=True, do_relu=True, relufactor=0.0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

    return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID",
                     name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")


        return conv


def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)


        return tf.nn.relu(out_res + inputres)


def build_generator_resnet_9blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6")
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7")
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8")
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9")

        o_c4 = general_deconv2d(o_r9, [batch_size, 256, 256, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv2d(o_c4, [batch_size, 512, 512, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c6 = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02, "SAME", "c6", do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6, "t1")


        return out_gen

def build_gen_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 1, 1, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 1, 1, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5



def patch_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f= 4

        patch_input = tf.random_crop(inputdisc,[1,70,70,3])
        o_c1 = general_conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5

with tf.variable_scope("Model") as scope:

    fake_B = build_generator_resnet_9blocks(input_A, name="g_A")
    fake_A = build_generator_resnet_9blocks(input_B, name="g_B")
    rec_A = build_gen_discriminator(input_A, "d_A")
    rec_B = build_gen_discriminator(input_B, "d_B")

    scope.reuse_variables()

    fake_rec_A = build_gen_discriminator(fake_A, "d_A")
    fake_rec_B = build_gen_discriminator(fake_B, "d_B")
    cyc_A = build_generator_resnet_9blocks(fake_B, "g_B")
    cyc_B = build_generator_resnet_9blocks(fake_A, "g_A")

    scope.reuse_variables()

    fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
    fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")

    img_A_summ = tf.summary.image("imageA", input_A)
    img_B_summ = tf.summary.image("imageb", input_B)
    fake_A_summ = tf.summary.image("fake_a", fake_A)
    fake_B_summ = tf.summary.image("fake_b", fake_B)
    rec_A_summ = tf.summary.image("rec_a", cyc_A)
    rec_B_summ = tf.summary.image("rec_b", cyc_B)




with tf.variable_scope("loss") as scope:
    cyc_loss = tf.reduce_mean(tf.abs(input_A - cyc_A)) + tf.reduce_mean(tf.abs(input_B - cyc_B))

    disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_rec_A, 1))
    disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_rec_B, 1))

    gt_A_and_A = tf.multiply(input_A, gt_A)
    gt_B_and_B = tf.multiply(input_B, gt_B)
    gt_A_and_fakeB = tf.multiply(fake_B, gt_A)
    gt_B_and_fakeA = tf.multiply(fake_A, gt_B)

    gt_loss_A = tf.reduce_mean(tf.abs(gt_A_and_A - gt_A_and_fakeB))
    gt_loss_B = tf.reduce_mean(tf.abs(gt_B_and_B - gt_B_and_fakeA))



    g_loss_A = cyc_loss * 10 + disc_loss_B+gt_loss_A*15
    g_loss_B = cyc_loss * 10 + disc_loss_A+gt_loss_B*15

    d_loss_A = (tf.reduce_mean(tf.square(fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(rec_A, 1))) / 2.0+gt_loss_A*10
    d_loss_B = (tf.reduce_mean(tf.square(fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(rec_B, 1))) / 2.0+gt_loss_B*10

    optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)

    model_vars = tf.trainable_variables()


    d_A_vars = [var for var in model_vars if 'd_A' in var.name]
    g_A_vars = [var for var in model_vars if 'g_A' in var.name]
    d_B_vars = [var for var in model_vars if 'd_B' in var.name]
    g_B_vars = [var for var in model_vars if 'g_B' in var.name]

    d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
    d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
    g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
    g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

    for var in model_vars: print(var.name)


    g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
    g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
    d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
    d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)



def fake_image_pool( num_fakes, fake, fake_pool):

    if(num_fakes < pool_size):
        fake_pool[num_fakes] = fake
        return fake
    else :
        p = random.random()
        if  p > 0.5:
            random_id = random.randint(0,pool_size-1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else :
            return fake

def save_training_images(sess, epoch):

    if not os.path.exists("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/breast/record"):
        os.makedirs("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/breast/record")

    for i in range(0,1):
        fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([fake_A, fake_B, cyc_A, cyc_B],feed_dict={input_A:A_input, input_B:B_input})
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/fake_a/fake_a_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/fake_b/fake_b_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/rec_a/rec_a_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/rec_b/rec_b_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/input_a/input_a_"+ str(epoch) + "_" + str(i)+".jpg",((A_input[0]+1)*127.5).astype(np.uint8))
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/input_b/input_b_"+ str(epoch) + "_" + str(i)+".jpg",((B_input[0]+1)*127.5).astype(np.uint8))

        A_img = np.concatenate((((A_input[0]+1)*127.5).astype(np.uint8), ((fake_B_temp[0]+1)*127.5).astype(np.uint8)), axis=1)
        B_img = np.concatenate((((B_input[0]+1)*127.5).astype(np.uint8), ((fake_A_temp[0]+1)*127.5).astype(np.uint8)), axis=1)
        img_result = np.concatenate((A_img, B_img), axis=0)
        imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/result/img_result"+ str(epoch) + "_" + str(i)+".jpg",img_result)



        #imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/cyclegan/IMG/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((A_input[0])).astype(np.uint8))
        #imsave("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/cyclegan/IMG/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((B_input[0])).astype(np.uint8))



filename_A='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/chase/tf record/chase.tfrecords'
filename_B='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/drive/data/train/tfrecord/drive.tfrecords'
writer = tf.summary.FileWriter("/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/record")
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1000)
max_images=1000
with tf.Session() as sess:
    sess.run(init)

    image_batch,label_A = createBatch(filename=filename_A, batchsize=1)
    label_batch,label_B = createBatch(filename=filename_B, batchsize=1)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(sess.run(global_step),200):
        print("In the epoch ", epoch)

        # Dealing with the learning rate as per the epoch number
        if (epoch < 100):
            curr_lr = 0.0002
        else:
            curr_lr = 0.0002 - 0.0002 * (epoch - 100) / 100

        for ptr in range(0, max_images):
            print("In the iteration ", ptr)
            print("Starting", time.time() * 1000.0)

            A_input,label_a,B_input,label_b=sess.run([image_batch,label_A,label_batch,label_B])

            # Optimizing the G_A network
            _,fake_B_temp, summary_str = sess.run([g_A_trainer, fake_B, g_A_loss_summ],feed_dict={input_A: A_input,input_B: B_input,gt_A:label_a,lr: curr_lr})
            writer.add_summary(summary_str, epoch * max_images + ptr)

            fake_B_temp1 = fake_image_pool(num_fake_inputs, fake_B_temp, fake_images_B)
            # Optimizing the D_B network
            _,summary_str = sess.run([d_B_trainer,d_B_loss_summ],feed_dict={input_A: A_input, input_B: B_input,gt_B:label_b,lr: curr_lr, fake_pool_B: fake_B_temp1})
            writer.add_summary(summary_str, epoch * max_images + ptr)
            # Optimizing the G_B network
            _,fake_A_temp, summary_str = sess.run([g_B_trainer, fake_A, g_B_loss_summ],
                                                   feed_dict={input_A: A_input,
                                                              input_B: B_input,gt_B:label_b, lr: curr_lr})
            writer.add_summary(summary_str, epoch * max_images + ptr)
            fake_A_temp1 = fake_image_pool(num_fake_inputs, fake_A_temp, fake_images_A)
            # Optimizing the D_A network
            _,summary_str = sess.run([d_A_trainer, d_A_loss_summ],
                                      feed_dict={input_A: A_input, input_B: B_input,gt_A:label_a,
                                                 lr: curr_lr, fake_pool_A: fake_A_temp1})
            writer.add_summary(summary_str, epoch * max_images + ptr)

            num_fake_inputs += 1
            if ptr % 100 == 0:
                saver.save(sess, "/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/model/model.ckpt",global_step=ptr)
            save_training_images(sess, epoch=ptr)
            G_loss_A = sess.run([g_loss_A],feed_dict={input_A: A_input, input_B: B_input,gt_A:label_a, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})
            G_loss_B = sess.run([g_loss_B],feed_dict={input_A: A_input, input_B: B_input,gt_B:label_b, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})
            D_loss_A = sess.run([d_loss_A],feed_dict={input_A: A_input, input_B: B_input, gt_A:label_a,lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})
            D_loss_B = sess.run([d_loss_B],feed_dict={input_A: A_input, input_B: B_input,gt_B:label_a, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})


            print('%d g_loss_A is %s,g_loss_A is %s,d_loss_A is %s,d_loss_b is %s'%(ptr,G_loss_A,G_loss_B,D_loss_A,D_loss_B))


        summary_str = sess.run(img_A_summ, feed_dict={input_A: A_input})
        writer.add_summary(summary_str, epoch * max_images + ptr)
        summary_str = sess.run(img_B_summ, feed_dict={input_B: B_input})
        writer.add_summary(summary_str, epoch * max_images + ptr)
        summary_str = sess.run(fake_A_summ, feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr})
        writer.add_summary(summary_str, epoch * max_images + ptr)
        summary_str = sess.run(fake_B_summ, feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr})
        writer.add_summary(summary_str, epoch * max_images + ptr)
        summary_str = sess.run(rec_A_summ,
                               feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr, fake_pool_B: fake_B_temp1})
        writer.add_summary(summary_str, epoch * max_images + ptr)
        summary_str = sess.run(rec_B_summ,
                               feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr, fake_pool_A: fake_A_temp1})
        writer.add_summary(summary_str, epoch * max_images + ptr)






        sess.run(tf.assign(global_step, epoch + 1))

        writer.add_graph(sess.graph)

    coord.request_stop()
    coord.join(threads)





