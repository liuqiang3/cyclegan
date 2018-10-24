import tensorflow as tf
import random
import time
import os
import numpy as np
from scipy.misc import imsave
from cyclegan_model import build_generator_resnet_9blocks,build_gen_discriminator
from cyclegan_read import createBatch
from cyclegan_layer import fake_image_pool
from skimage.measure import compare_ssim as SSIM

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



def train(filename_A='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/chase/tf record/chase.tfrecords',
          filename_B='/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/drive/data/train/tfrecord/drive.tfrecords',
          record = "/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/record",
          save_model="/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/Multi_center/cyc data/model/model.ckpt",
          save_img = "/media/root/4c3b1a1e-e2bc-42fe-aeff-321d032e5c70/tensorflow/breast/IMG/",
          save_time = 1,
          epochs=100000):
    batch_size = 1
    pool_size = 50

    img_height = 256
    img_width = 256
    img_layer = 3
    img_size = img_height * img_width
    input_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer], name="input_B")

    fake_pool_A = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="fake_pool_A")
    fake_pool_B = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name="fake_pool_B")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    num_fake_inputs = 0
    lr = tf.placeholder(tf.float32, shape=[], name="lr")
    fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layer))
    fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layer))


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
        identity_loss_B = tf.reduce_mean(tf.abs(input_A - fake_B))
        identity_loss_A = tf.reduce_mean(tf.abs(input_B - fake_A))
        disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_rec_A, 1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_rec_B, 1))



        g_loss_A = cyc_loss * 10 + disc_loss_B + identity_loss_B*3
        g_loss_B = cyc_loss * 10 + disc_loss_A + identity_loss_A*3

        # g_loss_A = cyc_loss * 10 + disc_loss_B + identity_loss*5
        # g_loss_B = cyc_loss * 10 + disc_loss_A + identity_loss*5

        d_loss_A = (tf.reduce_mean(tf.square(fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(rec_A, 1))) / 2.0
        d_loss_B = (tf.reduce_mean(tf.square(fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(rec_B, 1))) / 2.0

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

    def save_training_images(sess, epoch):
        save_image_path = save_img

        path_fake_a = save_image_path + 'fake_a/'
        if not os.path.exists(save_image_path + 'fake_a'):
            os.makedirs(save_image_path + 'fake_a')

        path_fake_b = save_image_path + 'fake_b/'
        if not os.path.exists(save_image_path + 'fake_b'):
            os.makedirs(save_image_path + 'fake_b')

        path_rec_a = save_image_path + 'rec_a/'
        if not os.path.exists(save_image_path + 'rec_a'):
            os.makedirs(save_image_path + 'rec_a')

        path_rec_b = save_image_path + 'rec_b/'
        if not os.path.exists(save_image_path + 'rec_b'):
            os.makedirs(save_image_path + 'rec_b')

        path_input_a = save_image_path + 'input_a/'
        if not os.path.exists(save_image_path + 'input_a'):
            os.makedirs(save_image_path + 'input_a')

        path_input_b = save_image_path + 'input_b/'
        if not os.path.exists(save_image_path + 'input_b'):
            os.makedirs(save_image_path + 'input_b')

        result_path = save_image_path + 'result/'
        if not os.path.exists(save_image_path + 'result'):
            os.makedirs(save_image_path + 'result')

        for i in range(0, save_time):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([fake_A, fake_B, cyc_A, cyc_B],
                                                                        feed_dict={input_A: A_input, input_B: B_input})
            imsave(path_fake_a + 'fake_a_' + str(
                epoch) + "_" + str(i) + ".jpg", ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave(path_fake_b + 'fake_b_' + str(
                epoch) + "_" + str(i) + ".jpg", ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave(path_rec_a + 'rec_a_' + str(
                epoch) + "_" + str(i) + ".jpg", ((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave(path_rec_b + 'rec_b_' + str(
                epoch) + "_" + str(i) + ".jpg", ((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave(path_input_a + 'input_a_' + str(
                epoch) + "_" + str(i) + ".jpg", ((A_input[0] + 1) * 127.5).astype(np.uint8))
            imsave(path_input_b + 'input_b_' + str(
                epoch) + "_" + str(i) + ".jpg", ((B_input[0] + 1) * 127.5).astype(np.uint8))

            A_img = np.concatenate(
                (((A_input[0] + 1) * 127.5).astype(np.uint8), ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8)), axis=1)
            B_img = np.concatenate(
                (((B_input[0] + 1) * 127.5).astype(np.uint8), ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8)), axis=1)
            img_result = np.concatenate((A_img, B_img), axis=0)
            imsave(result_path+'img_result' + str(
                epoch) + "_" + str(i) + ".jpg", img_result)

    filename_A=filename_A
    filename_B=filename_B
    writer = tf.summary.FileWriter(record)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1000)
    max_images=1000



    with tf.Session() as sess:
        sess.run(init)

        image_batch = createBatch(filename=filename_A, batchsize=1,img_width=img_width,img_height=img_height)
        label_batch = createBatch(filename=filename_B, batchsize=1,img_width=img_width,img_height=img_height)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(sess.run(global_step),epochs):
            print("In the epoch ", epoch)

            # Dealing with the learning rate as per the epoch number
            if (epoch < 100):
                curr_lr = 0.0002
            else:
                curr_lr = 0.0002 - 0.0002 * (epoch - 100) / 100

            for ptr in range(0, max_images):
                print("In the iteration ", ptr)
                print("Starting", time.time() * 1000.0)

                A_input,B_input=sess.run([image_batch,label_batch])

                # Optimizing the G_A network
                _,fake_B_temp, summary_str = sess.run([g_A_trainer, fake_B, g_A_loss_summ],feed_dict={input_A: A_input,input_B: B_input,lr: curr_lr})
                writer.add_summary(summary_str, epoch * max_images + ptr)

                fake_B_temp1 = fake_image_pool(num_fake_inputs, fake_B_temp, fake_images_B)
                # Optimizing the D_B network
                _,summary_str = sess.run([d_B_trainer,d_B_loss_summ],feed_dict={input_A: A_input, input_B: B_input,lr: curr_lr, fake_pool_B: fake_B_temp1})
                writer.add_summary(summary_str, epoch * max_images + ptr)
                # Optimizing the G_B network
                _,fake_A_temp, summary_str = sess.run([g_B_trainer, fake_A, g_B_loss_summ],
                                                       feed_dict={input_A: A_input,
                                                                  input_B: B_input, lr: curr_lr})
                writer.add_summary(summary_str, epoch * max_images + ptr)
                fake_A_temp1 = fake_image_pool(num_fake_inputs, fake_A_temp, fake_images_A)
                # Optimizing the D_A network
                _,summary_str = sess.run([d_A_trainer, d_A_loss_summ],
                                          feed_dict={input_A: A_input, input_B: B_input,
                                                     lr: curr_lr, fake_pool_A: fake_A_temp1})
                writer.add_summary(summary_str, epoch * max_images + ptr)

                num_fake_inputs += 1
                if ptr % 100 == 0:
                    saver.save(sess, save_model,global_step=ptr)
                save_training_images(sess, epoch=ptr)
                G_loss_A = sess.run([g_loss_A],feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})
                G_loss_B = sess.run([g_loss_B],feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})
                D_loss_A = sess.run([d_loss_A],feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})
                D_loss_B = sess.run([d_loss_B],feed_dict={input_A: A_input, input_B: B_input, lr: curr_lr,fake_pool_A:fake_A_temp1,fake_pool_B:fake_B_temp1})


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





